"""Local, catalog-aware prompt enhancement handler."""

from __future__ import annotations

import logging
import random
import uuid
from collections.abc import Generator, Callable, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING, Literal, NamedTuple

from _routes._errors import HTTPError
from api_types import (
    ConditioningType,
    EnhancePromptRequest,
    EnhancePromptResponse,
    IcLoraCatalogItem,
    LoraCatalogItem,
    LTXLocalModelId,
    LTXVideoGenAspectRatio,
    LTXVideoGenResolution,
    PromptEnhancerStatusResponse,
    PromptProvenance,
)
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from server_utils.media_validation import normalize_optional_path, validate_image_file
from services.analytics import resolution_for_provenance as shared_resolution_for_provenance
from services.analytics import prompt_provenance_label as shared_prompt_provenance_label
from services.features.ic_lora_recipes import ic_lora_skips_prompt_enhancement
from services.features.video.job_common import UNNAMED_RECIPE, denoise_canvas_for
from services.features.video.reference_sheet import ensure_reference_sheet
from services.gemini_text_client import resolve_gemini_model
from services.generation_interrupt import is_cancel_exception
from services.interfaces import PromptEnhancerPipeline
from services.lora_catalog import LoraCatalogProvider
from services.prompt_enhancement import (
    build_audio_visual_caption_system_prompt,
    build_conditioning_system_prompt,
    build_ic_lora_enhancement_system_prompt,
    build_image_edit_system_prompt,
    build_image_generation_system_prompt,
    build_keyframe_enhancement_system_prompt,
    build_lora_enhancement_system_prompt,
    build_template_fill_system_prompt,
    enforce_trigger_placements,
    fill_prompt_template,
    parse_template_fill_response,
)
from services.prompt_enhancement.i2v_frames import (
    KeyframeStill,
    SceneFrame,
    VideoRole,
    reference_sheet_still,
    reference_still,
    strip_panel_labels,
)
from services.prompt_enhancement.source_video import SourceVideo
from services.prompt_enhancement.video_frames import sampled_video_frames
from services.prompt_enhancer_pipeline.gemini_prompt_enhancer_pipeline import GeminiPromptEnhancerPipeline
from services.records import MediaError
from services.services_utils import get_device_type
from state.app_state_types import AppState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

# Manual Enhance is exploratory — a redo should offer a different rewrite even when generation
# seeds are locked — so it always draws fresh here rather than going through
# StateHandlerBase._resolve_seed(). Automatic enhancement is the opposite case; see
# _automatic_enhancement_seed().
_MAX_ENHANCE_SEED = 2147483647


class ResolvedGenerationPrompt(NamedTuple):
    """The prompt a local generation should submit, and how it got (or will get) enhanced."""

    prompt: str
    enhance_via_api: bool
    enhanced_locally: bool
    skip_recipe_wrap: bool = False

    @property
    def enhancement(self) -> str:
        """Where the rewrite happens, for run logs: local, server-side, or nowhere."""
        if self.enhanced_locally:
            return "local"
        return "api" if self.enhance_via_api else "none"


PromptProvenanceLabel = Literal["raw", "manually-enhanced", "auto-enhanced"]


def prompt_provenance_label(
    provenance: PromptProvenance,
    resolved: ResolvedGenerationPrompt,
) -> PromptProvenanceLabel:
    """How this run asked for the prompt to be passed. Not the prompt text."""
    return shared_prompt_provenance_label(
        provenance,
        enhanced_locally=resolved.enhanced_locally,
        enhance_via_api=resolved.enhance_via_api,
    )


def _video_role(req: EnhancePromptRequest) -> VideoRole:
    """A list of panel images is its own role. An IC-LoRA transforms its video, so the stills
    are ground truth. Anything else (Extend, Retake) only gets scene context from them and the
    typed prompt leads."""
    if req.imagePaths:
        return "panels"
    return "driving" if req.icLoraId is not None else "scene"


def _ic_lora_rewrite(ic_lora: IcLoraCatalogItem, run: Callable[[str], str]) -> str:
    """The catalog policy of an IC-LoRA rewrite, for every path that makes one.

    ``run`` sends one rewrite call with the given system prompt and returns the raw reply.
    A template entry gets the template fill. Any other entry gets a free rewrite with its
    trigger phrases enforced. ``ValueError`` means the reply did not fill the template.
    """
    template = ic_lora.prompt_template
    if template is not None:
        raw = run(build_template_fill_system_prompt(ic_lora))
        values = parse_template_fill_response(raw, set(template.placeholders))
        return fill_prompt_template(template, values)
    enhanced = run(build_ic_lora_enhancement_system_prompt(ic_lora))
    # Trigger enforcement on an empty rewrite would make a bare trigger word the prompt.
    if not enhanced.strip():
        return enhanced
    return enforce_trigger_placements(enhanced, [ic_lora])


def _writes_blank_prompt(source_video: SourceVideo | None) -> bool:
    return source_video is not None and source_video.writes_blank_prompt


def _reference_frame(ic_lora: IcLoraCatalogItem, path: str) -> SceneFrame:
    """A catalog entry pins the frame its look image conditions: the first frame of the clip
    (Restore), or a still before the clip (Layout to Render). An unpinned one uses frame 0."""
    pinned = ic_lora.reference_image_frame
    return reference_still(path, is_clip_frame=pinned is None or pinned >= 0)


class PromptEnhancementHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        lora_catalog_provider: LoraCatalogProvider,
        prompt_enhancer_pipeline_class: type[PromptEnhancerPipeline],
        gemini_pipeline: GeminiPromptEnhancerPipeline,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text_handler = text_handler
        self._lora_catalog_provider = lora_catalog_provider
        self._prompt_enhancer_pipeline_class = prompt_enhancer_pipeline_class
        self._gemini_pipeline = gemini_pipeline
        # One Gemini enhance at a time, manual or automatic, as the generation gate allowed.
        # It bounds the API cost of a Remote client that sends many requests. A manual
        # request that finds it taken gets a 409. An automatic rewrite waits for it.
        self._gemini_enhance_slot = Lock()

    def _random_seed(self) -> int:
        return random.randint(0, _MAX_ENHANCE_SEED)

    def status(self, *, prefer_gemini: bool) -> PromptEnhancerStatusResponse:
        """Flags for the Home Enhance control. ``prefer_gemini`` is Remote's default."""
        settings = self.state.app_settings
        has_gemini = bool(settings.gemini_api_key.strip())
        local_downloaded = (
            self._text_handler.resolve_prompt_enhancer_root_if_downloaded() is not None
        )
        has_local = local_downloaded and not self.config.force_api_generations
        if prefer_gemini:
            default_provider: Literal["local", "api"] = (
                "api" if has_gemini else "local" if has_local else "api"
            )
        else:
            preference = settings.prompt_enhancer_provider_preference
            if preference == "api":
                preferred: Literal["local", "api"] = "api"
            elif preference == "local" and has_local:
                preferred = "local"
            elif has_local:
                preferred = "local"
            else:
                preferred = "api"
            default_provider = (
                "local"
                if preferred == "api" and not has_gemini and has_local
                else preferred
            )
        return PromptEnhancerStatusResponse(
            exploreAutoEnhancePrompts=settings.explore_auto_enhance_prompts,
            hasGeminiApiKey=has_gemini,
            localEnhancementSupported=has_local,
            defaultProvider=default_provider,
            canToggleProvider=has_local and has_gemini,
            showManualEnhance=not settings.explore_auto_enhance_prompts,
        )

    def _automatic_enhancement_seed(
        self, generation_seed: int, *, explicit_generation_seed: bool
    ) -> int:
        """The seed an automatic rewrite runs on, given the seed the generation resolved to.

        Automatic enhancement is part of what produces the video: the model never sees the
        prompt as typed, only the rewrite. Drawing that rewrite independently would leave a
        pinned local generation irreproducible — same seed, different prompt, different clip —
        so an explicit request seed or seed lock makes the rewrite ride the generation's seed.
        An unlocked request without an explicit seed draws fresh like manual Enhance does.
        """
        if explicit_generation_seed or self.state.app_settings.seed_locked:
            return generation_seed
        return self._random_seed()

    def enhance(self, req: EnhancePromptRequest) -> EnhancePromptResponse:
        if req.provider == "api":
            return self._enhance_with_gemini(req)
        return self._enhance_with_gemma(req)

    def _enhance_with_gemini(self, req: EnhancePromptRequest) -> EnhancePromptResponse:
        """Manual Enhance on Gemini: an HTTP call that needs no GPU, so it runs beside a generation.

        It must not touch the generation slot. ``start_api_generation`` clears the Stop request
        of a running generation, and ``complete_generation`` or ``fail_generation`` would
        overwrite that generation's state.
        """
        if not self.state.app_settings.gemini_api_key.strip():
            raise HTTPError(400, "GEMINI_API_KEY_MISSING")
        if not self._gemini_enhance_slot.acquire(blocking=False):
            raise HTTPError(409, "Enhance already in progress")
        try:
            enhanced = self._resolve_and_enhance(req, None)
        except HTTPError:
            raise
        except Exception as e:
            raise HTTPError(500, str(e)) from e
        finally:
            self._gemini_enhance_slot.release()
        return EnhancePromptResponse(enhancedPrompt=enhanced)

    def _enhance_with_gemma(self, req: EnhancePromptRequest) -> EnhancePromptResponse:
        # Enhance never occupies the GPU slot (see PipelinesHandler.
        # evict_resident_pipeline) but still needs to mutually exclude with
        # generation and with itself — an abandoned/orphaned enhance call (e.g. the tab reloaded
        # mid-request) must not race a Generate click, a second Enhance click, or a generation
        # that's still loading its pipeline (reserved_generation_start covers that window; a bare
        # is_generation_running() check does not — see its own docstring). The "api" generation
        # slot gives us the mutual exclusion for free: it's the same bookkeeping every other
        # handler already does, and it doesn't require gpu_slot to be set.
        with self._generation.reserved_generation_start():
            gemma_root = self._text_handler.resolve_prompt_enhancer_root_if_downloaded()
            if gemma_root is None:
                raise HTTPError(409, "LOCAL_TEXT_ENCODER_NOT_AVAILABLE")

            generation_id = uuid.uuid4().hex[:8]
            self._generation.start_api_generation(generation_id)
            try:
                enhanced = self._resolve_and_enhance(req, gemma_root)
            except HTTPError as e:
                self._generation.fail_generation(e.detail)
                raise
            except Exception as e:
                self._generation.fail_generation(str(e))
                raise HTTPError(500, str(e)) from e

            self._generation.complete_generation(enhanced)
            return EnhancePromptResponse(enhancedPrompt=enhanced)

    def _resolve_and_enhance(self, req: EnhancePromptRequest, gemma_root: str | None) -> str:
        if self.config.local_generations_mode == "mac_mlx_q4":
            pipeline = self._pipelines.load_gpu_pipeline("fast").pipeline
            return pipeline.enhance_prompt(req.prompt, mode="i2v" if req.imagePath else "t2v", seed=self._random_seed())
        if req.imagePaths:
            return self._resolve_and_enhance_media(
                req,
                gemma_root,
                frames=self._sheet_frames(req.imagePaths, req.resolution, req.aspectRatio),
            )
        video_path = normalize_optional_path(req.videoPath)
        if video_path is None:
            return self._resolve_and_enhance_media(req, gemma_root, frames=None)
        window = req.replacedWindow
        source_video = SourceVideo(
            path=video_path,
            replaced_window=None if window is None else (window.start, window.end),
            # The request only has an image beside a video when it names an IC-LoRA.
            reference_image_path=(
                normalize_optional_path(req.imagePath) if req.icLoraId is not None else None
            ),
        )
        ic_lora = (
            None if req.icLoraId is None else self._lora_catalog_provider.get_ic_lora(req.icLoraId)
        )
        with self._video_frames(source_video, ic_lora) as frames:
            return self._resolve_and_enhance_media(req, gemma_root, frames=frames)

    def _sheet_frames(
        self,
        paths: Sequence[str],
        resolution: LTXVideoGenResolution | None = None,
        aspect_ratio: LTXVideoGenAspectRatio | None = None,
        *,
        sheet_path: str | None = None,
    ) -> list[SceneFrame]:
        """Build or reuse the reference sheet in ``.in-flight`` and return it as one frame."""
        for path in paths:
            # Before it reaches either provider: Gemini would otherwise get unchecked file bytes.
            validate_image_file(path)
        if sheet_path is not None and Path(sheet_path).is_file():
            return [reference_sheet_still(sheet_path)]
        canvas = denoise_canvas_for(
            resolution or "720p", aspect_ratio or "16:9", name=UNNAMED_RECIPE
        )
        try:
            sheet = ensure_reference_sheet(
                paths, *canvas, app_data_dir=self.config.app_data_dir
            )
        except (OSError, ValueError) as exc:
            raise HTTPError(400, "Invalid image file") from exc
        return [reference_sheet_still(str(sheet))]

    @contextmanager
    def _video_frames(
        self, source_video: SourceVideo, ic_lora: IcLoraCatalogItem | None
    ) -> Generator[list[SceneFrame]]:
        """Stills of the source video, after the IC-LoRA's look image if there is one.

        The files are removed when the context exits. The look image is the user's own file
        and stays. A look image needs ``ic_lora`` to say which frame it conditions. An unknown
        IC-LoRA id leaves it out here and is a 404 further on.
        """
        reference: SceneFrame | None = None
        if source_video.reference_image_path is not None and ic_lora is not None:
            if not ic_lora.allows_reference_image:
                raise HTTPError(400, "This IC-LoRA does not take a reference image")
            # Before it reaches either provider: Gemini would otherwise get unchecked file bytes.
            validate_image_file(source_video.reference_image_path)
            reference = _reference_frame(ic_lora, source_video.reference_image_path)
        try:
            with sampled_video_frames(
                source_video.path, replaced_window=source_video.replaced_window
            ) as frames:
                yield frames if reference is None else [reference, *frames]
        except MediaError as exc:
            # No path in the message: Remote error bodies must not carry filesystem paths.
            raise HTTPError(400, "Invalid video file") from exc

    def _resolve_and_enhance_media(
        self,
        req: EnhancePromptRequest,
        gemma_root: str | None,
        frames: Sequence[SceneFrame] | None,
    ) -> str:
        if req.mediaType == "image":
            # No catalog LoRA concept for images (validated at the request level) — always an
            # explicit, image-domain system prompt, never the video-oriented generic fallback.
            system_prompt = (
                build_image_edit_system_prompt() if req.imagePath is not None
                else build_image_generation_system_prompt()
            )
            return self._run_free_rewrite(req, system_prompt, gemma_root)

        if req.icLoraId is not None:
            ic_lora = self._lora_catalog_provider.get_ic_lora(req.icLoraId)
            if ic_lora is None:
                raise HTTPError(404, "LORA_CATALOG_ID_NOT_FOUND")
            return self._enhance_ic_lora(ic_lora, req, gemma_root, frames)

        if req.loraCatalogIds:
            loras: list[LoraCatalogItem] = []
            for catalog_id in req.loraCatalogIds:
                lora = self._lora_catalog_provider.get_lora(catalog_id)
                if lora is None:
                    raise HTTPError(404, "LORA_CATALOG_ID_NOT_FOUND")
                loras.append(lora)
            return self._enhance_loras(loras, req, gemma_root)

        if req.conditioningType is not None:
            system_prompt = build_conditioning_system_prompt(req.conditioningType)
            return self._run_free_rewrite(req, system_prompt, gemma_root)

        return self._run_free_rewrite(
            req, self._default_video_system_prompt(req, has_video=frames is not None), gemma_root, frames
        )

    def _default_video_system_prompt(
        self, req: EnhancePromptRequest, *, has_video: bool
    ) -> str | None:
        if req.keyframes:
            return self._keyframe_system_prompt()
        # A source video is a vision input like an image, so it takes the image-to-video style.
        return self._video_system_prompt(t2v=req.imagePath is None and not has_video)

    def _keyframe_system_prompt(self, model_id: LTXLocalModelId | None = None) -> str:
        spec = self._text_handler.ltx_model_spec(model_id)
        audio_visual = spec is not None and spec.wants_audio_visual_captions
        return build_keyframe_enhancement_system_prompt(audio_visual=audio_visual)

    def _video_system_prompt(
        self, *, t2v: bool, model_id: LTXLocalModelId | None = None
    ) -> str | None:
        """The model's own caption style, or None to keep each provider's default.

        Only the audio-visual generations (2.5) need this: their captions cover the soundscape,
        which neither the generic Gemini fallback nor a 2.3-era prompt asks for.
        """
        spec = self._text_handler.ltx_model_spec(model_id)
        if spec is None or not spec.wants_audio_visual_captions:
            return None
        return build_audio_visual_caption_system_prompt(t2v=t2v)

    def resolve_for_generation(
        self,
        prompt: str,
        *,
        provenance: PromptProvenance,
        generation_seed: int,
        explicit_generation_seed: bool = False,
        image_path: str | None = None,
        last_image_path: str | None = None,
        keyframes: list[KeyframeStill] | None = None,
        duration: int | None = None,
        fps: int | None = None,
        ic_lora: IcLoraCatalogItem | None = None,
        lora_catalog_ids: list[str] | None = None,
        conditioning_type: ConditioningType | None = None,
        local_model_id: LTXLocalModelId | None = None,
        explore_generation: bool = False,
        audio_path: str | None = None,
        source_video: SourceVideo | None = None,
        panel_paths: Sequence[str] = (),
    ) -> ResolvedGenerationPrompt:
        """The single enhancement policy for prompt-bearing local generation.

        Local encoding attempts a rewrite unless this is an Explore generation with
        auto-enhance off. API encoding forwards ``enhance_prompt`` according to the T2V/I2V
        settings (I2V and keyframe jobs use the I2V toggle; T2V, IC-LoRA, and A2V —
        including A2V with an optional start frame — use T2V). Three things still suppress
        both paths: an "enhanced" provenance, meaning the
        user already ran Enhance by hand and kept the result (rewriting a rewrite drifts the
        prompt away from what they approved); an empty prompt, which has nothing to expand (unless a source video lets
        the rewrite write one, see ``SourceVideo.writes_blank_prompt``); and Explore auto-enhance
        off, which sends the prompt as typed (Gen Space is unchanged).

        The two text-encoding paths enhance in different places: API encoding rewrites
        server-side inside the same /prompt-embedding call, so it only needs the flag forwarded.
        Local encoding has no such step, so the rewrite happens here — without it the model sees
        the prompt as typed, which for a version captioned in 150-220 word audio-visual
        paragraphs (2.5) lands far outside its training distribution and it invents the rest.

        ``lora_catalog_ids`` are the catalog ids of the plain LoRAs the request selected. They
        get the same catalog-aware treatment manual Enhance gives them, because the generic
        rewrite would paraphrase a required trigger phrase away and leave the adapter inert.

        ``generation_seed`` is the seed the caller already resolved for this run (an explicit
        request seed, the lock, or a fresh draw). It reaches the rewrite when
        ``explicit_generation_seed`` is true or seed locking is enabled — see
        _automatic_enhancement_seed() — and never reaches the API path, whose ``enhance_prompt``
        flag takes no seed.

        ``source_video`` is the source video of an IC-LoRA, Extend, or Retake job. The rewrite
        then sees a few stills of it, on local Gemma or, only when the user explicitly chose
        Gemini as the enhancer provider and stored a key, on Gemini. See ``SourceVideo`` for what
        it carries and ``enhance_for_generation`` for the rewrite.

        ``panel_paths`` are the ordered reference images of an image-list IC-LoRA (Ingredients).
        The rewrite sees them like a source video, with the same Gemini consent rule. A blank
        prompt is still not rewritten: the IC-LoRA create and run refuse it first.

        Must be called before the pipeline is loaded and before start_generation: the enhancer
        needs the VRAM a resident pipeline holds, and PipelinesHandler refuses to evict one
        while a generation is running.
        """
        if self.config.local_generations_mode == "mac_mlx_q4":
            if provenance == "enhanced" or not prompt.strip() or not self.state.app_settings.prompt_enhancer_enabled:
                return ResolvedGenerationPrompt(prompt, False, False)
            self._generation.raise_if_cancelled()
            try:
                pipeline = self._pipelines.load_gpu_pipeline("fast").pipeline
                enhanced = pipeline.enhance_prompt(prompt, mode="i2v" if image_path else "t2v", seed=generation_seed)
            except Exception:
                self._generation.raise_if_cancelled()
                logger.warning("MLX prompt enhancement failed; keeping the submitted prompt", exc_info=True)
                return ResolvedGenerationPrompt(prompt, False, False)
            self._generation.raise_if_cancelled()
            return ResolvedGenerationPrompt(enhanced, False, True)

        preview = self._preview_resolution(
            prompt,
            provenance=provenance,
            explore_generation=explore_generation,
            local_model_id=local_model_id,
            ic_lora=ic_lora,
            source_video=source_video,
            panel_paths=panel_paths,
        )
        if preview is not None:
            if preview.skip_recipe_wrap and provenance != "enhanced" and prompt.strip():
                logger.info(
                    "Explore auto-enhance off: sending prompt as typed (%d chars)",
                    len(prompt),
                )
            return preview
        enhanced, rewritten = self.enhance_for_generation(
            prompt,
            generation_seed=generation_seed,
            explicit_generation_seed=explicit_generation_seed,
            image_path=image_path,
            last_image_path=last_image_path,
            keyframes=keyframes,
            duration=duration,
            fps=fps,
            ic_lora=ic_lora,
            loras=self._resolve_generation_loras(lora_catalog_ids),
            conditioning_type=conditioning_type,
            local_model_id=local_model_id,
            source_video=source_video,
            panel_paths=panel_paths,
        )
        return ResolvedGenerationPrompt(enhanced, False, rewritten, False)

    def preview_prompt_provenance(
        self,
        prompt: str,
        *,
        provenance: PromptProvenance,
        explore_generation: bool = False,
        local_model_id: LTXLocalModelId | None = None,
        force_api: bool = False,
    ) -> PromptProvenanceLabel:
        """The analytics label for this request, without rewriting the prompt.

        A local rewrite that has not run yet counts as auto-enhanced. That is the
        mode this run requested; a later Gemma fallback still used that mode.
        """
        preview = self._preview_resolution(
            prompt,
            provenance=provenance,
            explore_generation=explore_generation,
            local_model_id=local_model_id,
            force_api=force_api,
        )
        if preview is None:
            preview = ResolvedGenerationPrompt(prompt, False, True)
        return shared_prompt_provenance_label(
            provenance,
            enhanced_locally=preview.enhanced_locally,
            enhance_via_api=preview.enhance_via_api,
        )

    def _preview_resolution(
        self,
        prompt: str,
        *,
        provenance: PromptProvenance,
        explore_generation: bool,
        local_model_id: LTXLocalModelId | None,
        force_api: bool = False,
        ic_lora: IcLoraCatalogItem | None = None,
        source_video: SourceVideo | None = None,
        panel_paths: Sequence[str] = (),
    ) -> ResolvedGenerationPrompt | None:
        """Resolution when no in-app rewrite is required.

        None means a rewrite will be attempted here: on local Gemma, or on Gemini for a source
        video when the user chose Gemini.
        """
        if explore_generation and ic_lora is not None and ic_lora_skips_prompt_enhancement(ic_lora.id):
            return ResolvedGenerationPrompt(prompt, False, False, False)
        decided = shared_resolution_for_provenance(
            provenance,
            prompt=prompt,
            explore_generation=explore_generation,
            explore_auto_enhance_prompts=self.state.app_settings.explore_auto_enhance_prompts,
            use_local_encoding=self._text_handler.should_use_local_encoding(local_model_id),
            prompt_enhancer_enabled=self.state.app_settings.prompt_enhancer_enabled,
            local_enhancer_available=(
                self._text_handler.resolve_prompt_enhancer_root_if_downloaded(local_model_id)
                is not None
            ),
            force_api=force_api,
            gemini_video_enhance=self._wants_gemini_media_enhance(source_video, panel_paths),
            enhance_blank_prompt=_writes_blank_prompt(source_video),
        )
        if decided is None:
            return None
        enhance_via_api, enhanced_locally, skip_recipe_wrap = decided
        return ResolvedGenerationPrompt(prompt, enhance_via_api, enhanced_locally, skip_recipe_wrap)

    def gemini_video_enhance_enabled(self) -> bool:
        """Whether an automatic rewrite of a source video or panel images may go to Gemini.

        Manual Enhance with Gemini is an explicit click. Automatic enhancement is not, and it
        would upload stills of the user's video or the user's panel images. So it needs an explicit provider choice (not
        the default that merely follows a stored key) as well as the key.
        """
        settings = self.state.app_settings
        return (
            settings.prompt_enhancer_provider_preference == "api"
            and bool(settings.gemini_api_key.strip())
        )

    def _wants_gemini_media_enhance(
        self, source_video: SourceVideo | None, panel_paths: Sequence[str]
    ) -> bool:
        return (source_video is not None or bool(panel_paths)) and self.gemini_video_enhance_enabled()

    def api_enhance_prompt(
        self,
        prompt: str,
        *,
        provenance: PromptProvenance,
        image_path: str | None = None,
        keyframes: list[KeyframeStill] | None = None,
        audio_path: str | None = None,
    ) -> bool:
        """Whether the LTX API should rewrite this prompt on the server.

        Used for API text encoding (``enhance_prompt`` on /prompt-embedding) and for
        fully-remote video (``enhance_prompt`` on /v2/text-to-video and siblings). The
        LTX gateway defaults that flag to true when omitted, so callers must send it.

        One setting covers every conditioning: the conditioning arguments stay in the
        signature because callers pass them, but they no longer select between gates.
        """
        if provenance == "enhanced" or not prompt.strip():
            return False
        return self.state.app_settings.prompt_enhancer_enabled

    def _resolve_generation_loras(self, catalog_ids: list[str] | None) -> list[LoraCatalogItem]:
        """The catalog entries behind a generation's selected plain LoRAs, by id alone.

        Unlike manual Enhance, an unresolvable id doesn't 404: automatic enhancement is a
        quality step layered onto a generation the user already started, so a stale id costs
        the catalog treatment rather than the run.
        """
        items: list[LoraCatalogItem] = []
        for catalog_id in catalog_ids or []:
            lora = self._lora_catalog_provider.get_lora(catalog_id)
            if lora is None:
                logger.info("Skipping unknown LoRA catalog id for enhancement: %s", catalog_id)
                continue
            # A templated entry needs the template-fill path, which the multi-select free
            # rewrite has no shape for. No plain catalog LoRA has one today.
            if lora.prompt_template is not None:
                continue
            items.append(lora)
        return items

    def enhance_for_generation(
        self,
        prompt: str,
        *,
        generation_seed: int,
        explicit_generation_seed: bool,
        image_path: str | None,
        last_image_path: str | None = None,
        keyframes: list[KeyframeStill] | None = None,
        duration: int | None = None,
        fps: int | None = None,
        ic_lora: IcLoraCatalogItem | None = None,
        loras: list[LoraCatalogItem] | None = None,
        conditioning_type: ConditioningType | None = None,
        local_model_id: LTXLocalModelId | None = None,
        source_video: SourceVideo | None = None,
        panel_paths: Sequence[str] = (),
    ) -> tuple[str, bool]:
        """Rewrite ``prompt`` on the local enhancer: ``(prompt, rewritten)``.

        With a ``source_video`` or ``panel_paths`` the rewrite sees stills of it or the panel
        images, and runs on Gemini instead of local Gemma when the user explicitly chose Gemini
        (see ``gemini_video_enhance_enabled``). Local Gemma stays the default. Without media, the
        rewrite is always local Gemma. An IC-LoRA job that has nothing to show (Gen Space, the
        Studio form) keeps its text-only rewrite. A Gemini rewrite waits for the one Gemini slot.

        ``rewritten`` is False on every fallback below, which is what lets a caller's run log
        distinguish "the enhancer rewrote this" from "the enhancer never ran".

        Home auto-enhance calls this whenever the enhancer checkpoint is on disk, including
        when text embeddings come from the API. A plain API-encoding run with no local
        checkpoint still enhances server-side and never gets here. Deliberately not
        `enhance()` — the caller already holds the generation slot, and the provider is not a
        request choice: it is local Gemma unless the settings send a source video to Gemini.

        Only cancellation is re-raised. Enhancement is otherwise a quality step, so a missing
        checkpoint or failed rewrite degrades to a deterministic fallback rather than failing
        the generation.
        """
        loras = loras or []
        if not prompt.strip() and not _writes_blank_prompt(source_video):
            return prompt, False
        gemma_root = self._text_handler.resolve_prompt_enhancer_root_if_downloaded(
            model_id=local_model_id
        )
        use_gemini = self._wants_gemini_media_enhance(source_video, panel_paths)
        if gemma_root is None and not use_gemini:
            logger.info("Skipping automatic enhancement: no local prompt enhancer downloaded")
            return self._generation_fallback_prompt(prompt, ic_lora, loras), False

        try:
            seed = self._automatic_enhancement_seed(
                generation_seed, explicit_generation_seed=explicit_generation_seed
            )
            if panel_paths:
                assert ic_lora is not None, "panel images come with an IC-LoRA"
                with self._gemini_enhance_slot if use_gemini else nullcontext():
                    enhanced = self._enhance_panels_for_generation(
                        prompt,
                        panel_paths,
                        ic_lora=ic_lora,
                        seed=seed,
                        provider="api" if use_gemini else "local",
                        gemma_root=gemma_root,
                        sheet_path=image_path,
                    )
            elif source_video is not None:
                with self._gemini_enhance_slot if use_gemini else nullcontext():
                    enhanced = self._enhance_video_for_generation(
                        prompt,
                        source_video,
                        ic_lora=ic_lora,
                        seed=seed,
                        provider="api" if use_gemini else "local",
                        gemma_root=gemma_root,
                        local_model_id=local_model_id,
                    )
            else:
                assert gemma_root is not None
                pipeline = self._load_prompt_enhancer_pipeline(gemma_root)
                if ic_lora is not None:
                    enhanced = _ic_lora_rewrite(
                        ic_lora,
                        lambda system_prompt: pipeline.enhance_t2v(
                            prompt, system_prompt=system_prompt, seed=seed
                        ),
                    )
                else:
                    system_prompt = self._generation_system_prompt(
                        loras=loras,
                        image_path=image_path,
                        keyframes=keyframes,
                        conditioning_type=conditioning_type,
                        local_model_id=local_model_id,
                    )
                    if image_path is not None or keyframes:
                        first_path = image_path or (keyframes[0][0] if keyframes else None)
                        assert first_path is not None
                        enhanced = pipeline.enhance_i2v(
                            prompt,
                            first_path,
                            system_prompt=system_prompt,
                            seed=seed,
                            last_image_path=None if keyframes else last_image_path,
                            keyframes=keyframes,
                            duration=duration,
                            fps=fps,
                        )
                    else:
                        enhanced = pipeline.enhance_t2v(prompt, system_prompt=system_prompt, seed=seed)
                    if enhanced.strip():
                        # Deterministic, same as manual Enhance: the rewrite is asked for the
                        # trigger phrases, never trusted to have kept them.
                        enhanced = enforce_trigger_placements(enhanced, loras)
        except Exception as exc:
            if is_cancel_exception(exc):
                raise
            logger.warning("Automatic local enhancement failed; using the fallback prompt", exc_info=True)
            return self._generation_fallback_prompt(prompt, ic_lora, loras), False

        if not enhanced.strip():
            return self._generation_fallback_prompt(prompt, ic_lora, loras), False
        logger.info(
            "Enhanced prompt locally for generation (%d -> %d chars)",
            len(prompt),
            len(enhanced),
        )
        return enhanced, True

    def _generation_fallback_prompt(
        self,
        prompt: str,
        ic_lora: IcLoraCatalogItem | None,
        loras: list[LoraCatalogItem],
    ) -> str:
        """Preserve every non-template catalog adapter's trigger when no rewrite is available."""
        items: list[LoraCatalogItem] = list(loras)
        if ic_lora is not None and ic_lora.prompt_template is None:
            items.append(ic_lora)
        return prompt if not items else enforce_trigger_placements(prompt, items)

    def _generation_system_prompt(
        self,
        *,
        loras: list[LoraCatalogItem],
        image_path: str | None,
        keyframes: list[KeyframeStill] | None,
        conditioning_type: ConditioningType | None,
        local_model_id: LTXLocalModelId | None = None,
    ) -> str | None:
        if loras:
            # Same treatment manual Enhance's multi-select path applies. It replaces the
            # model's own caption style rather than layering onto it: what the adapters need
            # from the prompt is the more specific constraint, and the LoRA block already
            # carries its own output-format instruction.
            return build_lora_enhancement_system_prompt(loras)
        if conditioning_type is not None:
            # Built-in canny/depth IC-LoRA: no catalog entry to draw from, but the same
            # "describe the reference scene faithfully" discipline manual Enhance applies.
            return build_conditioning_system_prompt(conditioning_type)
        if keyframes:
            return self._keyframe_system_prompt(model_id=local_model_id)
        return self._video_system_prompt(t2v=image_path is None, model_id=local_model_id)

    def _enhance_video_for_generation(
        self,
        prompt: str,
        source_video: SourceVideo,
        *,
        ic_lora: IcLoraCatalogItem | None,
        seed: int,
        provider: Literal["local", "api"],
        gemma_root: str | None,
        local_model_id: LTXLocalModelId | None,
    ) -> str:
        """The rewrite of an IC-LoRA, Extend, or Retake job that sees its source video.

        An IC-LoRA gets the same catalog policy manual Enhance gives it, with the stills as
        ground truth. Anything else (Extend, Retake) gets scene context only, so the typed
        prompt keeps the most weight. A template that can't be filled, or an unreadable video,
        raises out to the caller's typed-prompt fallback rather than failing the generation.
        """
        with self._video_frames(source_video, ic_lora) as frames:
            if ic_lora is None:
                return self._run_video_rewrite(
                    prompt,
                    frames,
                    self._video_system_prompt(t2v=False, model_id=local_model_id),
                    role="scene",
                    provider=provider,
                    gemma_root=gemma_root,
                    seed=seed,
                )
            return _ic_lora_rewrite(
                ic_lora,
                lambda system_prompt: self._run_video_rewrite(
                    prompt,
                    frames,
                    system_prompt,
                    role="driving",
                    provider=provider,
                    gemma_root=gemma_root,
                    seed=seed,
                ),
            )

    def _enhance_panels_for_generation(
        self,
        prompt: str,
        panel_paths: Sequence[str],
        *,
        ic_lora: IcLoraCatalogItem,
        seed: int,
        provider: Literal["local", "api"],
        gemma_root: str | None,
        sheet_path: str | None = None,
    ) -> str:
        """The rewrite of an image-list IC-LoRA job. It sees the constructed reference sheet.

        It uses the same catalog policy and reference sheet as manual Enhance. An unreadable image
        or a template that can't be filled raises out to the caller's typed-prompt fallback.
        """
        frames = self._sheet_frames(panel_paths, sheet_path=sheet_path)
        return _ic_lora_rewrite(
            ic_lora,
            lambda system_prompt: self._run_video_rewrite(
                prompt,
                frames,
                system_prompt,
                role="panels",
                provider=provider,
                gemma_root=gemma_root,
                seed=seed,
            ),
        )

    def _enhance_loras(
        self, loras: list[LoraCatalogItem], req: EnhancePromptRequest, gemma_root: str | None
    ) -> str:
        # None of the plain LoRAs in the catalog have a prompt_template today (only IC-LoRAs
        # do) — the multi-select path is always a free rewrite.
        system_prompt = build_lora_enhancement_system_prompt(loras)
        enhanced = self._run_free_rewrite(req, system_prompt, gemma_root)
        return enforce_trigger_placements(enhanced, loras)

    def _enhance_ic_lora(
        self,
        ic_lora: IcLoraCatalogItem,
        req: EnhancePromptRequest,
        gemma_root: str | None,
        frames: Sequence[SceneFrame] | None,
    ) -> str:
        if req.imagePaths and not ic_lora.is_sheet_recipe():
            raise HTTPError(422, "This IC-LoRA does not take a list of images")
        # A template scaffold has no image slot. Without a source video or a list of panel
        # images it is a text-only call.
        text_only = ic_lora.prompt_template is not None and frames is None
        target = (
            req.model_copy(update={"imagePath": None, "lastImagePath": None, "keyframes": []})
            if text_only
            else req
        )
        try:
            return _ic_lora_rewrite(
                ic_lora,
                lambda system_prompt: self._run_free_rewrite(
                    target, system_prompt, gemma_root, frames
                ),
            )
        except ValueError as e:
            raise HTTPError(500, f"PROMPT_TEMPLATE_FILL_FAILED: {e}") from e

    def _run_free_rewrite(
        self,
        req: EnhancePromptRequest,
        system_prompt: str | None,
        gemma_root: str | None,
        frames: Sequence[SceneFrame] | None = None,
    ) -> str:
        if frames is not None:
            return self._run_video_rewrite(
                req.prompt,
                frames,
                system_prompt,
                role=_video_role(req),
                provider=req.provider,
                gemma_root=gemma_root,
                seed=self._random_seed(),
            )
        # Reject an invalid/unreadable/oversized path before it reaches either provider — the
        # API path in particular would otherwise base64-encode and ship arbitrary file bytes to
        # a third-party API with no gate at all.
        keyframes = self._validated_keyframes(req)
        image_path = (
            None if keyframes else normalize_optional_path(req.imagePath)
        )
        last_image_path = (
            None
            if req.mediaType == "image" or keyframes
            else normalize_optional_path(req.lastImagePath)
        )
        if image_path is not None:
            validate_image_file(image_path)
        if last_image_path is not None:
            validate_image_file(last_image_path)

        seed = self._random_seed()
        first_path = image_path or (keyframes[0][0] if keyframes else None)
        if req.provider == "api":
            resolved_model = resolve_gemini_model(self.state.app_settings.gemini_model)
            logger.info("Enhancing prompt via Gemini API (%s)", resolved_model)
            api_key = self.state.app_settings.gemini_api_key.strip()
            if first_path is not None:
                return self._gemini_pipeline.enhance_i2v(
                    req.prompt,
                    first_path,
                    system_prompt=system_prompt,
                    seed=seed,
                    api_key=api_key,
                    model=resolved_model,
                    last_image_path=last_image_path,
                    keyframes=keyframes,
                    duration=req.duration,
                    fps=req.fps,
                )
            return self._gemini_pipeline.enhance_t2v(
                req.prompt,
                system_prompt=system_prompt,
                seed=seed,
                api_key=api_key,
                model=resolved_model,
            )

        logger.info("Enhancing prompt via local Gemma")
        assert gemma_root is not None
        pipeline = self._load_prompt_enhancer_pipeline(gemma_root)
        if first_path is not None:
            return pipeline.enhance_i2v(
                req.prompt,
                first_path,
                system_prompt=system_prompt,
                seed=seed,
                last_image_path=last_image_path,
                keyframes=keyframes,
                duration=req.duration,
                fps=req.fps,
            )
        return pipeline.enhance_t2v(req.prompt, system_prompt=system_prompt, seed=seed)

    def _validated_keyframes(self, req: EnhancePromptRequest) -> list[KeyframeStill] | None:
        if req.mediaType == "image" or not req.keyframes:
            return None
        frames: list[KeyframeStill] = []
        for keyframe in req.keyframes:
            path = normalize_optional_path(keyframe.imagePath)
            if path is None:
                raise HTTPError(400, "Each keyframe requires an image path")
            validate_image_file(path)
            frames.append((path, keyframe.frameIndex, keyframe.strength))
        frames.sort(key=lambda item: item[1])
        return frames

    def _run_video_rewrite(
        self,
        prompt: str,
        frames: Sequence[SceneFrame],
        system_prompt: str | None,
        *,
        role: VideoRole,
        provider: Literal["local", "api"],
        gemma_root: str | None,
        seed: int,
    ) -> str:
        """One rewrite call that sees the source video's stills, on either provider."""
        reply = self._call_video_rewrite(
            prompt, frames, system_prompt, role=role, provider=provider, gemma_root=gemma_root, seed=seed
        )
        # The panel labels are input only. A small model sometimes copies them into the reply.
        return strip_panel_labels(reply) if role == "panels" else reply

    def _call_video_rewrite(
        self,
        prompt: str,
        frames: Sequence[SceneFrame],
        system_prompt: str | None,
        *,
        role: VideoRole,
        provider: Literal["local", "api"],
        gemma_root: str | None,
        seed: int,
    ) -> str:
        look_images = sum(1 for frame in frames if frame.is_reference)
        logger.info(
            "Video rewrite: %d stills, %d look image(s), role=%s, provider=%s",
            len(frames) - look_images,
            look_images,
            role,
            provider,
        )
        if provider == "api":
            resolved_model = resolve_gemini_model(self.state.app_settings.gemini_model)
            logger.info("Enhancing prompt with video frames via Gemini API (%s)", resolved_model)
            return self._gemini_pipeline.enhance_with_video_frames(
                prompt,
                frames,
                system_prompt,
                seed,
                role,
                api_key=self.state.app_settings.gemini_api_key.strip(),
                model=resolved_model,
            )
        logger.info("Enhancing prompt with video frames via local Gemma")
        assert gemma_root is not None
        pipeline = self._load_prompt_enhancer_pipeline(gemma_root)
        return pipeline.enhance_with_video_frames(prompt, frames, system_prompt, seed, role)

    def _load_prompt_enhancer_pipeline(self, gemma_root: str) -> PromptEnhancerPipeline:
        self._pipelines.evict_resident_pipeline()
        device = get_device_type(self.config.device)
        return self._prompt_enhancer_pipeline_class.create(gemma_root, device)
