from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Collection
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal, cast

from pydantic import JsonValue, ValidationError

from api_types import (
    CreateIcLoraRecipeRequest,
    CreateIcLoraSheetRequest,
    CreateIcLoraTrackRequest,
    AudioToVideoParams,
    Asset,
    CreateAudioToVideoParams,
    CreateAudioToVideoRequest,
    CreateImageToVideoRequest,
    CreateLoraRecipeRequest,
    CreateExtendRequest,
    CreateRetakeRequest,
    CreateTextToVideoRequest,
    EditPreflightRequest,
    EditPreflightResponse,
    ExtendParams,
    Generation,
    GenerationProgressResponse,
    GenerationRisk,
    IcLoraRecipeStoredParams,
    LTXQueuedAspectRatio,
    LTXVideoGenFps,
    LTXVideoGenResolution,
    OfferingId,
    QueueEntry,
    QueueProgress,
    QueueSnapshot,
    RetakeParams,
)
from _routes._errors import HTTPError
from handlers.base import resolve_models_dir
from handlers.hf_auth_utils import optional_hf_token
from runtime_config.accelerator import accelerator_backend
from runtime_config.ic_lora_high_res import (
    HIGH_RES_NOT_ENABLED,
    HIGH_RES_ONLY_IC_LORAS,
    high_res_cell,
    high_res_source_large_enough,
)
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
)
from runtime_config.ltx_capabilities import LtxCapabilityFeature
from runtime_config.model_download_specs import local_aspect_ratios
from services.features.video.job_common import (
    high_res_unsupported_error,
    require_capable_local_model,
    stored_generation_risk,
)
from services.aspect_ratio import find_closest_aspect_ratio_option
from services.media_probe import MAX_VIDEO_BYTES, video_fps
from services.features.ic_lora_recipes import (
    IC_LORA_RECIPES,
    IcLoraRecipe,
    ic_lora_input_cap_message,
    resolve_ic_lora_recipe_create,
    resolve_synthesized_ic_lora_create,
)
from services.features.lora_recipes import RecipeCreateError, resolve_recipe_create
from services.lora_catalog import LoraCatalogProvider
from services.store import Store
from services.sqlite_store import collect_asset_ids
from services.records import (
    AttemptError,
    AssetRecord,
    CapabilityFailedError,
    GenerationRecord,
    MediaError,
    QueueGenerationRecord,
    StatusError,
    UnavailableError,
)
from services.generation_queue.control import (
    QueueControl,
    NoOpQueueControl,
    cancel_queued_generation,
)
from services.generation_queue.finished_tray import FinishedTray
from services.generation_queue.registry import ExecutorRegistry
from services.features.video import (
    AUDIO_TO_VIDEO_FEATURE,
    EXTEND_FEATURE,
    FEATURE as TEXT_TO_VIDEO_FEATURE,
    IMAGE_TO_VIDEO_FEATURE,
    RETAKE_FEATURE,
)
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


AnalyticsClient = Literal["desktop", "remote"]
# Set only when the user did not ask for the job directly; analytics reports those as "user".
AnalyticsRequestedBy = Literal["assistant"]


logger = logging.getLogger(__name__)

# Recipe ids (the generation `feature`) of the catalog IC-LoRAs that need highResVideoToVideo.
_HIGH_RES_ONLY_RECIPE_IDS = frozenset(
    recipe.recipe_id
    for recipe in IC_LORA_RECIPES.values()
    if recipe.catalog_id in HIGH_RES_ONLY_IC_LORAS
)


def _generation_to_api(record: GenerationRecord) -> Generation:
    payload = record.model_dump(mode="json")
    spec = dict(record.spec)
    spec.pop("_analytics", None)
    spec.pop("_risk", None)
    payload["spec"] = spec
    # Junk fails closed on read the same way execute does. A strict validate
    # of the raw dict raises on list, get, queue, retry, and cancel.
    risk = stored_generation_risk(record.spec)
    payload["risk"] = None if risk is None else risk.model_dump(mode="json")
    return Generation.model_validate(payload)


# Create-time Fast budget gate. Raises HTTPError; returns the bet annotation.
QueuedFastPreflight = Callable[..., GenerationRisk | None]


# Create-time A2V budget gate over derived frames. Raises HTTPError; returns the bet annotation.
QueuedA2VPreflight = Callable[..., GenerationRisk | None]


# Create-time edit window + budget gate. Raises HTTPError; returns the bet annotation.
QueuedEditPreflight = Callable[..., GenerationRisk | None]


# Create-time IC-LoRA recipe budget gate. Raises HTTPError; returns the bet annotation.
QueuedIcLoraRecipePreflight = Callable[..., GenerationRisk | None]


# Advisory per-cell verdicts for edit pickers. Per-cell failures are returned, never raised.
EditMatrixPreflight = Callable[..., EditPreflightResponse]


def _ic_lora_create_error(exc: RecipeCreateError) -> HTTPError:
    if exc.code == "IC_LORA_NOT_DOWNLOADED":
        status = 409
    elif exc.code == "UNKNOWN_DOWNLOAD_VARIANT":
        status = 404
    else:
        status = 422
    return HTTPError(status, str(exc), code=exc.code)


class QueuedGenerationHandler:
    def __init__(
        self,
        store: Store,
        executor_registry: ExecutorRegistry,
        derive_local_a2v: Callable[[CreateAudioToVideoParams, int], AudioToVideoParams],
        queue_control: QueueControl | None = None,
        *,
        progress_reader: Callable[[], GenerationProgressResponse] | None = None,
        config: "RuntimeConfig | None" = None,
        lora_catalog_provider: LoraCatalogProvider | None = None,
        state: AppState | None = None,
        lock: RLock | None = None,
        preflight_queued_fast: QueuedFastPreflight,
        preflight_queued_a2v: QueuedA2VPreflight,
        preflight_queued_edit: QueuedEditPreflight,
        preflight_edit_matrix: EditMatrixPreflight,
        preflight_queued_ic_lora_recipe: QueuedIcLoraRecipePreflight,
        preflight_queued_synthesized_ic_lora: QueuedIcLoraRecipePreflight,
        high_res_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._db = store
        self._executor_registry = executor_registry
        self._derive_local_a2v = derive_local_a2v
        self._preflight_queued_fast = preflight_queued_fast
        self._preflight_queued_a2v = preflight_queued_a2v
        self._preflight_queued_edit = preflight_queued_edit
        self._preflight_edit_matrix = preflight_edit_matrix
        self._preflight_queued_ic_lora_recipe = preflight_queued_ic_lora_recipe
        self._preflight_queued_synthesized_ic_lora = preflight_queued_synthesized_ic_lora
        self._queue_control = (
            NoOpQueueControl() if queue_control is None else queue_control
        )
        self._progress_reader = progress_reader
        # LoRA-recipe create needs the models dir, the catalog, and HF-auth state to
        # resolve a catalog id to an installed path and to gate typed misses before
        # enqueue. Optional so tests that only exercise the plain t2v/i2v create paths
        # (or construct the handler directly) don't have to wire them.
        self._config = config
        self._lora_catalog_provider = lora_catalog_provider
        self._state = state
        self._lock = lock
        self._high_res_enabled = high_res_enabled or (lambda: False)
        self._finished = FinishedTray()

    def recover_on_boot(self) -> None:
        self._db.fail_running_on_boot()

    def create_text_to_video(
        self,
        req: CreateTextToVideoRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        spec: dict[str, JsonValue] = {
            "params": req.params.model_dump(mode="json"),
            "inputs": {},
        }
        risk = self._preflight_fast_params(
            offering=req.params.model,
            resolution=req.params.resolution,
            aspect_ratio=req.params.aspectRatio,
            duration=req.params.duration,
            fps=req.params.fps,
        )
        return self._enqueue_generation(
            TEXT_TO_VIDEO_FEATURE,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
        )

    def create_image_to_video(
        self,
        req: CreateImageToVideoRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        start_frame = self._require_image_asset(req.inputs.startFrame.assetId, slot="startFrame")
        if req.inputs.endFrame is not None:
            self._require_image_asset(req.inputs.endFrame.assetId, slot="endFrame")
        spec: dict[str, JsonValue] = {
            "params": req.params.model_dump(mode="json"),
            "inputs": req.inputs.model_dump(mode="json", exclude_none=True),
        }
        risk = self._preflight_fast_params(
            offering=req.params.model,
            resolution=req.params.resolution,
            aspect_ratio=self._resolve_queued_preflight_aspect(
                req.params.aspectRatio, req.params.resolution, start_frame
            ),
            duration=req.params.duration,
            fps=req.params.fps,
        )
        return self._enqueue_generation(
            IMAGE_TO_VIDEO_FEATURE,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
        )

    def create_lora_recipe(
        self,
        recipe_id: str,
        req: CreateLoraRecipeRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        """Create a queued generation for a LoRA recipe.

        The client sends the catalog id + scale, never a filesystem path.
        :func:`resolve_recipe_create` validates the recipe (unknown / mode / device /
        HF / visibility / not-downloaded / missing start frame) and stores
        ``loras: [{ ref: "", scale, catalogId }]``. Execute hydrates ``ref`` via
        ``make_recipe_lora_resolver``. Anything invalid raises a typed miss *before*
        this handler enqueues — no row, no GPU slot. The row is stored under
        ``feature == recipe_id`` (the web Explore id). i2v recipes also persist
        ``inputs.startFrame``.
        """
        self._require_local_generation_possible()
        try:
            feature, params, inputs = resolve_recipe_create(
                recipe_id,
                req,
                catalog=self._require_catalog(),
                models_dir=self._models_dir(),
                device=self._recipe_device(),
                hf_authenticated=self._hf_authenticated(),
            )
        except RecipeCreateError as exc:
            raise HTTPError(422, str(exc), code=exc.code) from exc

        if inputs is not None:
            start_frame = self._require_image_asset(
                inputs.startFrame.assetId, slot="startFrame"
            )
            if inputs.endFrame is not None:
                self._require_image_asset(inputs.endFrame.assetId, slot="endFrame")
        else:
            start_frame = None
        # Same Fast budget gate as the plain t2v/i2v creates: the recipe's params
        # carry the client's offering, and execute budgets via that same field,
        # so preflighting here admits exactly what the run budgets. i2v auto
        # resolves from the start frame; t2v never sends auto.
        risk = self._preflight_fast_params(
            offering=params.model,
            resolution=params.resolution,
            aspect_ratio=self._resolve_queued_preflight_aspect(
                params.aspectRatio, params.resolution, start_frame
            ),
            duration=params.duration,
            fps=params.fps,
        )

        spec: dict[str, JsonValue] = {
            "params": params.model_dump(mode="json"),
            "inputs": (
                inputs.model_dump(mode="json", exclude_none=True) if inputs is not None else {}
            ),
        }
        return self._enqueue_generation(
            feature,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
        )

    def _require_local_generation_possible(self) -> None:
        # The Explore queue is local-only (no API fallback), so a machine that can't
        # generate locally must fail the create rather than enqueue a job that can
        # never run. Mirrors StateHandlerBase._require_local_generation_possible.
        if (
            self._config is not None
            and self._config.local_generations_mode == "unsupported"
        ):
            raise HTTPError(
                422,
                LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
                code=LOCAL_GENERATION_UNSUPPORTED,
            )

    def _recipe_device(self) -> str | None:
        # The real accelerator backend (cuda / rocm / mps / cpu), not the Darwin flag:
        # an Intel Mac is not MPS and a CPU/ROCm Linux box is not CUDA. None when there
        # is no config to gate against (tests that construct the handler directly).
        if self._config is None:
            return None
        return accelerator_backend()

    def _hf_authenticated(self) -> bool:
        if self._state is None or self._lock is None:
            return False
        return optional_hf_token(self._state, self._lock) is not None

    def _require_catalog(self) -> LoraCatalogProvider:
        if self._lora_catalog_provider is None:
            raise HTTPError(500, "LoRA catalog is not configured")
        return self._lora_catalog_provider

    def _models_dir(self) -> Path:
        if self._config is None:
            raise HTTPError(500, "Runtime config is not configured")
        return resolve_models_dir(self._state, self._config)

    def create_audio_to_video(
        self,
        req: CreateAudioToVideoRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        if req.params.prompt.strip() == "" and req.inputs.startFrame is None:
            raise HTTPError(
                422,
                "Connect a prompt or a start image before running this node.",
                code="INVALID_GENERATION_SPEC",
            )
        audio = self._require_audio_asset(req.inputs.audio.assetId, slot="audio")
        if audio.metadata.mediaType != "audio":
            raise HTTPError(422, "INVALID_AUDIO_ASSET", code="INVALID_GENERATION_SPEC")
        # Resolved before the budget runs: auto needs the start frame's shape,
        # and the budget needs the resolved aspect.
        start_frame = (
            None
            if req.inputs.startFrame is None
            else self._require_image_asset(
                req.inputs.startFrame.assetId, slot="startFrame"
            )
        )
        params = self._derive_local_a2v(
            req.params,
            audio.metadata.metadata.durationMs,
        )
        # budget_shape, the same canvas as a fully dropped row. Auto resolves
        # from the start frame first; a missing frame falls back to 16:9.
        risk = self._preflight_queued_a2v(
            offering=params.model,
            resolution=params.resolution,
            aspect_ratio=self._resolve_queued_preflight_aspect(
                params.aspectRatio, params.resolution, start_frame, is_a2v=True
            ),
            num_frames=params.numFrames,
        )
        spec: dict[str, JsonValue] = {
            "params": params.model_dump(mode="json"),
            "inputs": req.inputs.model_dump(mode="json", exclude_none=True),
        }
        return self._enqueue_generation(
            AUDIO_TO_VIDEO_FEATURE,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
        )

    def create_retake(
        self,
        req: CreateRetakeRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        return self._enqueue_video_edit(
            RETAKE_FEATURE,
            req,
            "retake",
            code="UNSUPPORTED_RETAKE",
            message="Retake is not supported for the selected LTX model.",
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            extra=lambda duration_ms: self._validate_retake_selection(
                req.params.startTime,
                req.params.duration,
                duration_ms,
            ),
        )

    def create_ic_lora_recipe(
        self,
        recipe_id: str,
        req: CreateIcLoraRecipeRequest | CreateIcLoraSheetRequest | CreateIcLoraTrackRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        if not isinstance(req, CreateIcLoraRecipeRequest):
            return self._create_synthesized_ic_lora(
                recipe_id, req, analytics_client=analytics_client, analytics_requested_by=analytics_requested_by,
                generation_id=generation_id,
            )
        self._require_local_generation_possible()
        high_res = high_res_cell(req.params.resolution)
        if high_res is not None and not self._high_res_enabled():
            raise HTTPError(422, HIGH_RES_NOT_ENABLED, code=HIGH_RES_NOT_ENABLED)
        video = self._require_asset(
            req.inputs.video.assetId,
            slot="video",
            kind="video",
            max_bytes=MAX_VIDEO_BYTES,
        )
        if video.metadata.mediaType != "video":
            raise HTTPError(422, "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC")
        if req.inputs.image is not None:
            self._require_image_asset(req.inputs.image.assetId, slot="image")
        source_fps = video.metadata.metadata.fps
        if source_fps is None:
            try:
                source_fps = video_fps(Path(video.path))
            except MediaError as exc:
                raise HTTPError(
                    422, "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC"
                ) from exc
        try:
            recipe, item, stored = resolve_ic_lora_recipe_create(
                recipe_id,
                req,
                catalog=self._require_catalog(),
                models_dir=self._models_dir(),
                hf_authenticated=self._hf_authenticated(),
                source_fps=source_fps,
            )
        except RecipeCreateError as exc:
            raise _ic_lora_create_error(exc) from exc
        if item.id in HIGH_RES_ONLY_IC_LORAS and not self._high_res_enabled():
            raise HTTPError(422, HIGH_RES_NOT_ENABLED, code=HIGH_RES_NOT_ENABLED)
        # A high cell runs one tiled stage. An IC-LoRA that keeps its LoRA on stage 2 has no
        # such run, and its look still cannot go through the tiled path.
        if high_res is not None and item.default_settings.stage_mode is IcLoraStageMode.STAGE_2_LORA:
            raise high_res_unsupported_error(item.name)
        too_long = ic_lora_input_cap_message(
            video.metadata.metadata.durationMs, stored.fps
        )
        if too_long is not None:
            raise HTTPError(422, too_long, code="IC_LORA_INPUT_TOO_LONG")
        self._require_offering_capability(
            req.params.model,
            "ic_lora",
            code="UNSUPPORTED_IC_LORA",
            message=f"{item.name} is not supported for the selected LTX model.",
        )
        # An upscaling recipe (Refine Details) scales a smaller clip up to the cell.
        if (
            high_res is not None
            and not recipe.upscales_to_cell
            and not high_res_source_large_enough(
                video.metadata.metadata.width, video.metadata.metadata.height, high_res
            )
        ):
            raise HTTPError(
                422,
                f"The clip is smaller than {req.params.resolution}. Pick a lower resolution "
                "or a larger clip.",
                code="IC_LORA_SOURCE_TOO_SMALL",
            )
        risk = self._preflight_queued_ic_lora_recipe(
            item=item,
            video=video,
            stored=stored,
            source_fps=source_fps,
            upscale=recipe.upscales_to_cell,
            cutout=recipe.cutout,
        )
        return self._enqueue_ic_lora_recipe(
            recipe,
            stored,
            req,
            risk=risk,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
        )

    def _create_synthesized_ic_lora(
        self,
        recipe_id: str,
        req: CreateIcLoraSheetRequest | CreateIcLoraTrackRequest,
        *,
        analytics_client: AnalyticsClient | None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        """Queue a run whose control clip the job makes. Its size and length come from the stored params."""
        self._require_local_generation_possible()
        try:
            recipe, item, stored = resolve_synthesized_ic_lora_create(
                recipe_id,
                req,
                catalog=self._require_catalog(),
                models_dir=self._models_dir(),
                hf_authenticated=self._hf_authenticated(),
                require_image=self._require_image_size,
            )
        except RecipeCreateError as exc:
            raise _ic_lora_create_error(exc) from exc
        self._require_offering_capability(
            req.params.model,
            "ic_lora",
            code="UNSUPPORTED_IC_LORA",
            message=f"{item.name} is not supported for the selected LTX model.",
        )
        return self._enqueue_ic_lora_recipe(
            recipe,
            stored,
            req,
            risk=self._preflight_queued_synthesized_ic_lora(recipe=recipe, item=item, stored=stored),
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
        )

    def _enqueue_ic_lora_recipe(
        self,
        recipe: IcLoraRecipe,
        stored: IcLoraRecipeStoredParams,
        req: CreateIcLoraRecipeRequest | CreateIcLoraSheetRequest | CreateIcLoraTrackRequest,
        *,
        risk: GenerationRisk | None,
        analytics_client: AnalyticsClient | None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        spec: dict[str, JsonValue] = {
            "params": stored.model_dump(mode="json", exclude={"scale", "variantId"}),
            "inputs": req.inputs.model_dump(mode="json", exclude_none=True),
        }
        return self._enqueue_generation(
            recipe.recipe_id,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
        )

    def create_extend(
        self,
        req: CreateExtendRequest,
        *,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        return self._enqueue_video_edit(
            EXTEND_FEATURE,
            req,
            "extend",
            code="UNSUPPORTED_EXTEND",
            message="Extend is not supported for the selected LTX model.",
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
        )

    def preflight_edit(self, req: EditPreflightRequest) -> EditPreflightResponse:
        """Advisory picker verdicts. Same asset checks as create (missing asset,
        wrong kind, oversized file fail identically); per-cell failures are
        returned, never raised, so the UI can disable with tooltips."""
        video = self._require_asset(
            req.assetId,
            slot="video",
            kind="video",
            max_bytes=MAX_VIDEO_BYTES,
        )
        if video.metadata.mediaType != "video":
            raise HTTPError(422, "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC")
        return self._preflight_edit_matrix(
            feature=req.feature,
            asset_path=video.path,
            start_time=req.startTime,
            mode=req.mode,
            candidates=req.candidates,
            durations=req.durations,
        )

    def _enqueue_video_edit(
        self,
        feature: str,
        req: CreateRetakeRequest | CreateExtendRequest,
        capability: LtxCapabilityFeature,
        *,
        code: str,
        message: str,
        analytics_client: AnalyticsClient | None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        extra: Callable[[int], None] | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        video = self._require_asset(
            req.inputs.video.assetId,
            slot="video",
            kind="video",
            max_bytes=MAX_VIDEO_BYTES,
        )
        if video.metadata.mediaType != "video":
            raise HTTPError(422, "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC")
        if extra is not None:
            extra(video.metadata.metadata.durationMs)
        self._require_offering_capability(
            req.params.model, capability, code=code, message=message
        )
        risk = self._preflight_edit_params(
            feature, asset_path=video.path, params=req.params
        )
        params_payload = req.params.model_dump(mode="json")
        spec: dict[str, JsonValue] = {
            "params": params_payload,
            "inputs": req.inputs.model_dump(mode="json", exclude_none=True),
        }
        return self._enqueue_generation(
            feature,
            spec,
            contract_version=req.contract_version,
            analytics_client=analytics_client,
            analytics_requested_by=analytics_requested_by,
            generation_id=generation_id,
            risk=risk,
        )

    def _preflight_fast_params(
        self,
        *,
        offering: OfferingId,
        resolution: LTXVideoGenResolution,
        aspect_ratio: LTXQueuedAspectRatio,
        duration: int | None,
        fps: LTXVideoGenFps,
    ) -> GenerationRisk | None:
        return self._preflight_queued_fast(
            offering=offering,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            fps=fps,
        )

    @staticmethod
    def _resolve_queued_preflight_aspect(
        aspect_ratio: LTXQueuedAspectRatio,
        resolution: LTXVideoGenResolution,
        start_frame: AssetRecord | None,
        *,
        is_a2v: bool = False,
    ) -> LTXQueuedAspectRatio:
        """Explicit aspect passes through; auto resolves from the start frame.

        Same closest-match execute uses, so create admits exactly what the run
        budgets. Corrupt image metadata 422s like its audio sibling. A missing
        frame falls back to the anchor rather than refusing the queue insert —
        the A2V no-start default; unreachable for t2v/i2v (i2v creates require
        one, t2v rejects auto).
        """
        if aspect_ratio != "auto":
            return aspect_ratio
        if start_frame is None:
            return "16:9"
        metadata = start_frame.metadata
        if metadata.mediaType != "image":
            raise HTTPError(422, "INVALID_IMAGE_ASSET", code="INVALID_GENERATION_SPEC")
        # find_closest returns an element of options, so the cast is exact.
        return cast(
            LTXQueuedAspectRatio,
            find_closest_aspect_ratio_option(
                metadata.metadata.width,
                metadata.metadata.height,
                local_aspect_ratios(resolution, is_a2v=is_a2v),
            ),
        )

    def _preflight_edit_params(
        self,
        feature: str,
        *,
        asset_path: str,
        params: RetakeParams | ExtendParams,
    ) -> GenerationRisk | None:
        return self._preflight_queued_edit(
            feature=feature,
            asset_path=asset_path,
            params=params,
        )

    def _require_asset(
        self,
        asset_id: str,
        *,
        slot: str,
        kind: Literal["image", "audio", "video"],
        max_bytes: int | None = None,
    ) -> AssetRecord:
        try:
            asset = self._db.get_asset(asset_id)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        missing = (
            f"{slot} image is no longer available"
            if kind == "image"
            else f"{slot} is no longer available"
        )
        if asset is None:
            raise HTTPError(
                422,
                missing,
                code="INPUT_ASSET_NOT_FOUND",
            )
        if asset.media_kind != kind:
            article = "an" if kind[0] in "aeiou" else "a"
            raise HTTPError(
                422,
                f"{slot} must reference {article} {kind} asset",
                code="INVALID_GENERATION_SPEC",
            )
        if max_bytes is not None:
            try:
                if Path(asset.path).stat().st_size > max_bytes:
                    raise HTTPError(400, "FILE_TOO_LARGE", code="FILE_TOO_LARGE")
            except OSError as exc:
                raise HTTPError(
                    422,
                    missing,
                    code="INPUT_ASSET_NOT_FOUND",
                ) from exc
        return asset

    def _require_image_asset(self, asset_id: str, *, slot: str) -> AssetRecord:
        return self._require_asset(asset_id, slot=slot, kind="image")

    def _require_image_size(self, asset_id: str) -> tuple[int, int]:
        image = self._require_image_asset(asset_id, slot="image")
        assert image.metadata.mediaType == "image"
        return image.metadata.metadata.width, image.metadata.metadata.height

    def _require_audio_asset(self, asset_id: str, *, slot: str) -> AssetRecord:
        return self._require_asset(asset_id, slot=slot, kind="audio")

    def _validate_retake_selection(
        self, start_time: float, duration: float, duration_ms: int
    ) -> None:
        source_seconds = duration_ms / 1000
        if source_seconds + 0.1 < 2.0:
            raise HTTPError(
                422,
                "This video is shorter than 2s. Replace it to continue.",
                code="INVALID_GENERATION_SPEC",
            )
        if start_time + duration > source_seconds + 0.1:
            raise HTTPError(
                422,
                "Selection is outside the usable video range",
                code="INVALID_GENERATION_SPEC",
            )

    def _require_offering_capability(
        self,
        offering: OfferingId,
        capability: LtxCapabilityFeature,
        *,
        code: str,
        message: str,
    ) -> None:
        try:
            require_capable_local_model(
                self._models_dir(),
                offering,
                capability,
                unsupported_code=code,
                unsupported_message=message,
            )
        except CapabilityFailedError as exc:
            raise HTTPError(409, exc.detail, code=exc.code) from exc

    def _enqueue_generation(
        self,
        feature: str,
        spec: dict[str, JsonValue],
        *,
        contract_version: int,
        analytics_client: AnalyticsClient | None = None,
        analytics_requested_by: AnalyticsRequestedBy | None = None,
        generation_id: str | None = None,
        risk: GenerationRisk | None = None,
    ) -> Generation:
        try:
            self._executor_registry.validate_params(
                feature, spec, contract_version=contract_version
            )
        except (ValidationError, ValueError, CapabilityFailedError) as exc:
            if isinstance(exc, CapabilityFailedError):
                raise HTTPError(
                    422,
                    exc.detail,
                    code=exc.code or "INVALID_GENERATION_SPEC",
                ) from exc
            raise HTTPError(
                422,
                str(exc),
                code="INVALID_GENERATION_SPEC",
            ) from exc
        if risk is not None:
            spec = {**spec, "_risk": risk.model_dump(mode="json")}
        if analytics_client is not None:
            analytics: dict[str, JsonValue] = {"client": analytics_client}
            if analytics_requested_by is not None:
                analytics["requested_by"] = analytics_requested_by
            spec = {**spec, "_analytics": analytics}
        try:
            record = self._db.insert_generation(
                feature,
                spec,
                contract_version=contract_version,
                generation_id=generation_id,
            )
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except ValueError as exc:
            raise HTTPError(400, str(exc)) from exc
        except sqlite3.IntegrityError as exc:
            raise HTTPError(400, str(exc)) from exc
        self._queue_control.wake()
        return _generation_to_api(record)

    def retry_generation(self, generation_id: str) -> Generation:
        try:
            record = self._db.retry_generation(generation_id)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except KeyError as exc:
            raise HTTPError(404, "GENERATION_NOT_FOUND") from exc
        except StatusError as exc:
            raise HTTPError(409, "INVALID_GENERATION_STATUS") from exc
        self._finished.release(generation_id)
        self._queue_control.wake()
        return _generation_to_api(record)

    def cancel_generation(self, generation_id: str) -> Generation:
        try:
            record = cancel_queued_generation(
                self._db,
                generation_id,
                self._queue_control.interrupt,
            )
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except KeyError as exc:
            raise HTTPError(404, "GENERATION_NOT_FOUND") from exc
        except (AttemptError, StatusError) as exc:
            raise HTTPError(409, "INVALID_GENERATION_STATUS") from exc
        self._queue_control.wake()
        return _generation_to_api(record)

    def get_queue_snapshot(self) -> QueueSnapshot:
        try:
            entries = self._db.list_queue_generations()
            self._finished.observe(entries, self._db.get_generation)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc

        active_record = next(
            (
                entry
                for entry in entries
                if entry.generation.status in {"running", "cancelling"}
            ),
            None,
        )
        active = (
            self._queue_entry(active_record, self._active_progress(active_record))
            if active_record is not None
            else None
        )
        queued = [
            self._queue_entry(entry, None)
            for entry in entries
            if entry.generation.status == "queued"
        ]
        done, failed, unseen_ids = self._finished.lists()
        return QueueSnapshot(
            active=active,
            queued=queued,
            done=[self._queue_entry(entry, None) for entry in done],
            failed=[self._queue_entry(entry, None) for entry in failed],
            unseen_ids=unseen_ids,
        )

    def remember_finished(self, generation_id: str) -> None:
        try:
            record = self._db.get_generation(generation_id)
            if record is None:
                return
            assets = tuple(
                asset
                for asset_id in collect_asset_ids(record.spec)
                if (asset := self._db.get_asset(asset_id)) is not None
            )
        except UnavailableError:
            return
        self._finished.remember(
            QueueGenerationRecord(generation=record, input_assets=assets)
        )

    def mark_done_seen(self, generation_id: str) -> QueueSnapshot:
        if not self._finished.mark_seen(generation_id):
            raise HTTPError(404, "GENERATION_NOT_FOUND")
        return self.get_queue_snapshot()

    def dismiss_done(self, generation_id: str) -> QueueSnapshot:
        if not self._finished.dismiss_done(generation_id):
            raise HTTPError(404, "GENERATION_NOT_FOUND")
        return self.get_queue_snapshot()

    def clear_done(
        self, *, feature_is_allowed: Callable[[str], bool] | None = None
    ) -> QueueSnapshot:
        self._finished.clear_done(feature_is_allowed)
        return self.get_queue_snapshot()

    def clear_failed(
        self, *, feature_is_allowed: Callable[[str], bool] | None = None
    ) -> QueueSnapshot:
        self._finished.clear_failed(feature_is_allowed)
        return self.get_queue_snapshot()

    def reorder_queue(
        self, generation_id: str, before_generation_id: str | None
    ) -> QueueSnapshot:
        try:
            self._db.reorder_queued_generation(generation_id, before_generation_id)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except KeyError as exc:
            raise HTTPError(404, "GENERATION_NOT_FOUND") from exc
        except StatusError as exc:
            raise HTTPError(409, "INVALID_GENERATION_STATUS") from exc
        self._queue_control.wake()
        return self.get_queue_snapshot()

    def _active_progress(
        self, active: QueueGenerationRecord
    ) -> QueueProgress | None:
        if self._progress_reader is None:
            return None
        progress = self._progress_reader()
        if progress.id != active.generation.id:
            return None
        return QueueProgress(
            phase=progress.phase,
            progress=progress.progress,
            currentStep=progress.currentStep,
            totalSteps=progress.totalSteps,
        )

    @staticmethod
    def _queue_entry(
        entry: QueueGenerationRecord, progress: QueueProgress | None
    ) -> QueueEntry:
        return QueueEntry(
            generation=_generation_to_api(entry.generation),
            input_assets=[
                Asset.model_validate(asset.model_dump(mode="json"))
                for asset in entry.input_assets
            ],
            progress=progress,
        )

    def list_generations(self, feature: str | None = None, *, limit: int | None = None) -> list[Generation]:
        try:
            records = self._db.list_generations(feature, limit=limit)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        return [_generation_to_api(record) for record in records]

    def list_recent_features(
        self, *, limit: int, allowed: Collection[str] | None = None
    ) -> list[str]:
        if allowed is not None and not self._high_res_enabled():
            # Drop the hidden recipes before the store applies the limit, so they do
            # not take a slot from an older feature that is offered.
            allowed = [name for name in allowed if name not in _HIGH_RES_ONLY_RECIPE_IDS]
        try:
            return self._db.list_recent_features(limit=limit, allowed=allowed)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc

    def get_generation(self, generation_id: str) -> Generation:
        try:
            record = self._db.get_generation(generation_id)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        if record is None:
            raise HTTPError(404, "GENERATION_NOT_FOUND")
        return _generation_to_api(record)

    def delete_generation(self, generation_id: str) -> None:
        try:
            deleted = self._db.delete_generation(generation_id)
        except UnavailableError as exc:
            raise HTTPError(503, "STORE_UNAVAILABLE") from exc
        except StatusError as exc:
            raise HTTPError(409, "INVALID_GENERATION_STATUS") from exc
        if not deleted:
            raise HTTPError(404, "GENERATION_NOT_FOUND")
        self._finished.forget(generation_id)
