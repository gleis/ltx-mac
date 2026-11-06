"""Video generation orchestration handler."""

from __future__ import annotations

import logging
import os
import subprocess

from collections.abc import Callable
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from PIL import Image

from frame_math import (
    AutoDurationSpec,
    compute_num_frames,
    effective_a2v_audio_seconds,
    num_frames_for_audio_duration,
    snap_up_to_multiple,
)
from api_types import (
    AudioToVideoParams,
    CreateAudioToVideoParams,
    EstimateVideoRequest,
    EstimateVideoResponse,
    GenerationRisk,
    OfferingId,
    GenerateVideoCancelledResponse,
    GenerateVideoCompleteResponse,
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    ImageConditioningInput,
    ReferenceImageInput, ReferenceImageRole,
    LoraEntry,
    LTXLocalModelId,
    LTXVideoGenerationModelSpecItem,
    LTXLocalA2VResolution,
    LTXQueuedAspectRatio,
    LTXVideoGenFps,
    LTXVideoGenResolution,
    PipelineOffloadMode,
    VideoCameraMotion,
    VideoLoadMode,
    WallTimeEstimateResponse,
)
from runtime_config.eta import (
    EtaFeature,
    EtaLoadMode,
    WallTimeEstimate,
    estimate_fast_wall_time_s,
    is_measured_hardware_class,
)
from runtime_config.evict_retry import evict_retry_filter
from runtime_config.ltx_capabilities import (
    CanvasMode,
    LtxAspectRatio,
    a2v_pixels_for,
    budget_shape,
    local_canvas,
    local_caps,
    pixels_for,
    supports,
)
from runtime_config.low_performance_machine import detect_low_performance_machine
from runtime_config.models_scanner import resolve_lora_ref
from runtime_config.runtime_policy import LocalGenerationMode
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    LocalJobFilter,
    VideoJobLoad,
    VideoJobReject,
    reject_message_and_code,
    risk_for_reject,
    risk_if_allowed,
    unmeasured_load_risk,
    vram_snapshot_suffix,
)
from _routes._errors import HTTPError
from api_model_specs import (
    FORCED_API_MODEL_MAP,
    build_generate_video_model_specs_response,
    forced_api_resolution_map,
    get_local_video_generation_model_specs,
    supported_duration_range,
    validate_generate_video_request,
)
from handlers.base import StateHandlerBase
from server_utils.heartbeat import log_heartbeat
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.prompt_enhancement_handler import PromptEnhancementHandler
from handlers.text_handler import TextHandler
from runtime_config.offerings import OFFERING_LOCAL_MODEL_IDS, resolve_offering_local_model_id
from runtime_config.model_download_specs import get_existing_cp_path, get_latest_ltx_model_id, is_duration_head_ready, resolve_active_ltx_model_id
from server_utils.media_validation import (
    normalize_optional_path,
    validate_audio_file,
    validate_image_file,
)
from server_utils.oriented_image import open_oriented_rgb, oriented_image_file
from services.denoising_progress import distilled_total_steps
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.interfaces import GpuInfo, LTXAPIClient
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from services.ltx_api_client.ltx_api_errors import map_ltx_api_client_error, missing_ltx_api_key_error
from services.patches import diffusion_stage_cache
from state.app_state_types import AppState
from state.app_settings import should_video_generate_with_ltx_api

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


def _upload_oriented_image(client: LTXAPIClient, *, api_key: str, path: Path) -> str:
    with oriented_image_file(path) as upload_path:
        return client.upload_file(api_key=api_key, file_path=str(upload_path))


def _lora_catalog_ids(loras: list[LoraEntry]) -> list[str]:
    """The catalog ids among the selected LoRAs, for catalog-aware prompt enhancement.

    A custom LoRA carries none and stays weights-only: its filename is deliberately not
    matched against the catalog, which would attach a stranger's trigger phrase to it.
    """
    return [entry.catalogId for entry in loras if entry.catalogId is not None]


FORCED_API_RESOLUTION_MAP: dict[str, dict[str, dict[str, str]]] = forced_api_resolution_map()


def _a2v_fps_durations(
    item: LTXVideoGenerationModelSpecItem | None,
    resolution: LTXLocalA2VResolution,
    fps: LTXVideoGenFps,
) -> list[int]:
    duration_map = None if item is None else item.spec.a2v_supported_resolutions_durations
    if duration_map is None:
        return []
    resolution_spec = duration_map.get(resolution)
    if resolution_spec is None:
        return []
    return list(resolution_spec.fps_to_durations.get(fps, []))


class VideoGenerationHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        prompt_enhancement_handler: PromptEnhancementHandler,
        ltx_api_client: LTXAPIClient,
        config: RuntimeConfig,
        gpu_info: GpuInfo,
    ) -> None:
        super().__init__(state, lock, config, gpu_info=gpu_info)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._prompt_enhancement = prompt_enhancement_handler
        self._ltx_api_client = ltx_api_client

    def _active_ltx_model_id(self) -> LTXLocalModelId | None:
        return resolve_active_ltx_model_id(
            self.models_dir, self.state.app_settings.active_ltx_model_id, self.config.local_generations_mode
        )

    def _duration_head_ready_for(self, model_id: LTXLocalModelId | None) -> bool:
        return model_id is not None and is_duration_head_ready(self.models_dir, model_id)

    def _duration_head_ready(self) -> bool:
        return self._duration_head_ready_for(self._active_ltx_model_id())

    def _local_pixels(
        self,
        resolution: LTXVideoGenResolution,
        aspect: LtxAspectRatio,
        *,
        invalid_code: str,
        local_model_id: LTXLocalModelId | None = None,
        mode: CanvasMode = "video",
    ) -> tuple[int, int]:
        model_id = local_model_id or self._active_ltx_model_id()
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        lookup = a2v_pixels_for if mode == "a2v" else pixels_for
        try:
            return lookup(local_caps(model_id), resolution, aspect)
        except KeyError as exc:
            raise HTTPError(400, invalid_code) from exc

    def get_model_specs(self) -> GenerateVideoModelsSpecsResponse:
        # Display on total: the picker lists every total-fitting cell; GenSpace
        # refuses over-free cells at click, queued jobs re-check at execute.
        job_filter = self._local_job_filter(total_only=True)
        return build_generate_video_model_specs_response(
            local_model_id=job_filter.profile.model_id,
            models_dir=self.models_dir,
            duration_head_ready=self._duration_head_ready(),
            job_filter=job_filter,
            low_performance_machine=detect_low_performance_machine(self.config),
        )

    def estimate_local_fast(self, req: EstimateVideoRequest) -> EstimateVideoResponse:
        """Pre-generate wall-time estimate for one local Fast cell (ETA v1).

        Budgets exactly like generate (same 422s), then answers from the table.
        Null estimate means unknown — unlisted cell, off-class hardware, or
        automatic duration — never an error.
        """
        if req.duration is None:
            return EstimateVideoResponse(estimate=None)
        canvas_mode: CanvasMode = "a2v" if req.feature == "a2v" else "video"
        job_filter = self._local_job_filter()
        # Local models share one pixel grid, so the latest id is exact here
        # when none is installed.
        budget_width, budget_height = budget_shape(
            local_caps(job_filter.profile.model_id or get_latest_ltx_model_id()),
            req.resolution,
            req.aspectRatio,
            mode=canvas_mode,
        )
        frames = compute_num_frames(req.duration, req.fps)
        load_mode, risk = self._video_job_load_mode(
            budget_width, budget_height, frames, job_filter=job_filter
        )
        est = self._estimate_wall_time(
            budget_width, budget_height, frames, load_mode, req.feature
        )
        if est is None:
            return EstimateVideoResponse(estimate=None, risk=risk)
        return EstimateVideoResponse(
            estimate=WallTimeEstimateResponse(
                seconds=est.seconds, basis=est.basis, t2vReuse=est.t2v_reuse
            ),
            risk=risk,
        )

    def derive_local_a2v_params(
        self, params: CreateAudioToVideoParams, audio_duration_ms: int
    ) -> AudioToVideoParams:
        num_frames = self._local_a2v_num_frames(
            resolution=params.resolution,
            model=params.model,
            fps=params.fps,
            aspect_ratio=params.aspectRatio,
            audio_duration_ms=audio_duration_ms,
        )
        return AudioToVideoParams(
            **params.model_dump(mode="python"),
            numFrames=num_frames,
        )

    def validate_local_a2v_request(
        self, params: AudioToVideoParams, audio_duration_ms: int
    ) -> None:
        num_frames = self._local_a2v_num_frames(
            resolution=params.resolution,
            model=params.model,
            fps=params.fps,
            aspect_ratio=params.aspectRatio,
            audio_duration_ms=audio_duration_ms,
        )
        if params.numFrames != num_frames:
            raise HTTPError(
                422,
                "Audio-to-video frame count no longer matches the selected audio.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )

    def _fast_local_spec_item(
        self,
        model_id: LTXLocalModelId,
        *,
        job_filter: LocalJobFilter | None,
    ) -> LTXVideoGenerationModelSpecItem | None:
        return next(
            (
                candidate
                for candidate in get_local_video_generation_model_specs(
                    model_id,
                    duration_head_ready=self._duration_head_ready_for(model_id),
                    job_filter=job_filter,
                )
                if candidate.pipeline == "fast"
            ),
            None,
        )

    def _prefix_a2v_audio_seconds(
        self,
        *,
        model_id: LTXLocalModelId,
        resolution: LTXVideoGenResolution,
        fps: LTXVideoGenFps,
        audio_duration_seconds: float,
    ) -> float | None:
        """Opening slice this resolution will hear.

        A clip past this cell but inside a longer one is cut to the cell max.
        A clip past every cell is None, so a caller cannot treat it as the
        whole clip. Missing specs leave the length unchanged. The cell is the
        static ceiling: a budget drop is not a shorter cell, or a kept-max
        would silently trim audio the budget gate should refuse or annotate.
        """
        item = self._fast_local_spec_item(model_id, job_filter=None)
        duration_map = None if item is None else item.spec.a2v_supported_resolutions_durations
        if duration_map is None:
            return audio_duration_seconds
        resolution_spec = duration_map.get(resolution)
        if resolution_spec is None:
            return audio_duration_seconds
        durations = resolution_spec.fps_to_durations.get(fps, [])
        if not durations:
            return audio_duration_seconds
        longest = 0
        for cell in duration_map.values():
            cell_durations = cell.fps_to_durations.get(fps, [])
            if cell_durations:
                longest = max(longest, max(cell_durations))
        return effective_a2v_audio_seconds(
            audio_duration_seconds,
            cell_max_seconds=float(max(durations)),
            longest_cell_seconds=float(longest),
        )

    def _local_a2v_num_frames(
        self,
        *,
        resolution: LTXLocalA2VResolution,
        model: OfferingId,
        fps: LTXVideoGenFps,
        aspect_ratio: LTXQueuedAspectRatio,
        audio_duration_ms: int,
    ) -> int:
        # Envelope of the checkpoint that will actually run, not the catalog
        # preferred row (2.3 1.1 while only 1.0 is on disk).
        a2v_model_id = resolve_offering_local_model_id(self.models_dir, model, self.config.local_generations_mode)
        if a2v_model_id is None:
            raise HTTPError(409, "LTX_MODEL_NOT_INSTALLED")
        # Shape comes from the static ceiling. The budget filter's kept max is
        # not a shorter cell: audio inside the ceiling but past the kept max is
        # a dropped duration, priced below, not "audio is too long".
        static_item = self._fast_local_spec_item(a2v_model_id, job_filter=None)
        if static_item is None or static_item.spec.a2v_supported_resolutions_durations is None:
            raise HTTPError(
                422,
                "Audio-to-video is not available for the selected offering.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        resolution_spec = static_item.spec.a2v_supported_resolutions_durations.get(resolution)
        if resolution_spec is None:
            raise HTTPError(
                422,
                "Audio-to-video is not available at the selected resolution.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        durations = resolution_spec.fps_to_durations.get(fps, [])
        if not durations:
            raise HTTPError(
                422,
                "Audio-to-video is not available at the selected frame rate and duration.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        audio_duration_seconds = audio_duration_ms / 1000
        if audio_duration_seconds < 2.0:
            raise HTTPError(
                422,
                "Audio is too short for audio-to-video.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        max_duration = max(durations)
        heard = self._prefix_a2v_audio_seconds(
            model_id=a2v_model_id,
            resolution=resolution,
            fps=fps,
            audio_duration_seconds=audio_duration_seconds,
        )
        if heard is None or heard >= max_duration + 0.1:
            raise HTTPError(
                422,
                "Audio is too long for audio-to-video.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        max_frames = compute_num_frames(max_duration, fps)
        num_frames = num_frames_for_audio_duration(
            heard,
            fps,
            max_frames=max_frames,
        )
        if num_frames is None:
            raise HTTPError(
                422,
                "Audio is too long for audio-to-video.",
                code="INVALID_VIDEO_GENERATION_SPEC",
            )
        # Total, like queued create: create and execute agree on the shape.
        # Free fit is the execute gate's problem.
        job_filter = self._local_job_filter(model_id=a2v_model_id, total_only=True)
        filtered_item = self._fast_local_spec_item(a2v_model_id, job_filter=job_filter)
        kept = _a2v_fps_durations(filtered_item, resolution, fps)
        if not kept:
            # The picker dropped the whole row. That stays a hard refusal in
            # every mode: there is no warned cell to accept. A derived shape
            # that still loads is quoted as the shortest static cell.
            self._reject_budget_dropped_a2v_cell(
                job_filter,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                num_frames=num_frames,
                static_cell_frames=compute_num_frames(min(durations), fps),
            )
        return num_frames

    def generate(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        self._require_local_generation_possible(api_key=self.state.app_settings.ltx_api_key)
        use_api_specs = should_video_generate_with_ltx_api(
            force_api_generations=self.config.force_api_generations,
            settings=self.state.app_settings,
        )
        self._validate_generate_video_request(req, use_api_specs=use_api_specs)

        if use_api_specs:
            return self._generate_forced_api(req)

        with self._generation.reserved_generation_start():
            # Release dead pools before the budget samples free: the queue
            # runner does this pre-execute, and start_generation evicts only
            # after the sample. Best-effort, like the runner's.
            try:
                diffusion_stage_cache.evict()
            except Exception:
                logger.warning(
                    "Direct generate reclaim before sample failed; continuing",
                    exc_info=True,
                )
            return self._generate_local_reserved_validated(
                req, generation_id=self._make_generation_id()
            )

    def generate_local_reserved(
        self,
        req: GenerateVideoRequest,
        *,
        generation_id: str,
        output_path: Path | None = None,
        prompt_wrap: Callable[[str], str] | None = None,
        a2v_num_frames: int | None = None,
        a2v_audio_duration_seconds: float | None = None,
        local_model_id: LTXLocalModelId,
        explore_generation: bool = True,
        risk_allowance: bool | None = None,
        unmeasured_bet: bool = False,
    ) -> GenerateVideoResponse:
        # Local-only Home/Remote entry: checkpoint is required. GenSpace
        # ``generate()`` still uses Settings via ``_generate_local_reserved_validated``.
        self._require_local_generation_possible()
        self._validate_generate_video_request(
            req, use_api_specs=False, local_model_id=local_model_id
        )
        return self._generate_local_reserved_validated(
            req,
            generation_id=generation_id,
            output_path=output_path,
            prompt_wrap=prompt_wrap,
            a2v_num_frames=a2v_num_frames,
            a2v_audio_duration_seconds=a2v_audio_duration_seconds,
            local_model_id=local_model_id,
            explore_generation=explore_generation,
            risk_allowance=risk_allowance,
            unmeasured_bet=unmeasured_bet,
        )

    def _validate_generate_video_request(
        self,
        req: GenerateVideoRequest,
        *,
        use_api_specs: bool,
        local_model_id: LTXLocalModelId | None = None,
    ) -> None:
        job_model_id = None if use_api_specs else (local_model_id or self._active_ltx_model_id())
        validation_error = validate_generate_video_request(
            req,
            use_api_specs=use_api_specs,
            local_model_id=job_model_id,
            duration_head_ready=False if use_api_specs else self._duration_head_ready_for(job_model_id),
        )
        if validation_error is not None:
            raise HTTPError(422, validation_error, code="INVALID_VIDEO_GENERATION_SPEC")

    def _generate_local_reserved_validated(
        self,
        req: GenerateVideoRequest,
        *,
        generation_id: str,
        output_path: Path | None = None,
        prompt_wrap: Callable[[str], str] | None = None,
        a2v_num_frames: int | None = None,
        a2v_audio_duration_seconds: float | None = None,
        local_model_id: LTXLocalModelId | None = None,
        explore_generation: bool = False,
        risk_allowance: bool | None = None,
        unmeasured_bet: bool = False,
    ) -> GenerateVideoResponse:
        resolution = req.resolution
        duration = req.duration
        fps = req.fps
        job_model_id = local_model_id or self._active_ltx_model_id()
        if self.config.local_generations_mode == "mac_mlx_q4" and job_model_id is not None:
            if not job_model_id.startswith("ltx-2.3"):
                raise HTTPError(422, "This model is not supported by the pinned MLX runtime")
            job_model_id = "ltx-2.3-mlx-q4"

        audio_path = normalize_optional_path(req.audioPath)
        if audio_path:
            if self.config.local_generations_mode == "mac_mlx_q4":
                raise HTTPError(400, "LOCAL_MAC_MLX_A2V_NOT_SUPPORTED_IN_FIRST_MILESTONE")
            if duration is None:
                raise HTTPError(
                    422,
                    "Automatic duration cannot be combined with audio-to-video",
                    code="INVALID_VIDEO_GENERATION_SPEC",
                )
            return self._generate_a2v(
                req,
                duration,
                fps,
                audio_path=audio_path,
                generation_id=generation_id,
                output_path=output_path,
                num_frames=a2v_num_frames,
                audio_duration_seconds=a2v_audio_duration_seconds,
                local_model_id=job_model_id,
                explore_generation=explore_generation,
                risk_allowance=risk_allowance,
                unmeasured_bet=unmeasured_bet,
            )

        logger.info("Resolution %s - using fast pipeline", resolution)

        width, height = self._local_pixels(
            resolution,
            req.aspectRatio,
            invalid_code="INVALID_LOCAL_RESOLUTION",
            local_model_id=job_model_id,
        )
        width = snap_up_to_multiple(width, 64)
        height = snap_up_to_multiple(height, 64)

        job_filter = self._local_job_filter(model_id=job_model_id)
        if duration is None:
            item = next(
                candidate
                for candidate in get_local_video_generation_model_specs(
                    job_model_id,
                    duration_head_ready=self._duration_head_ready_for(job_model_id),
                    job_filter=job_filter,
                )
                if candidate.pipeline == req.model
            )
            try:
                min_seconds, max_seconds = supported_duration_range(
                    item, resolution=resolution, fps=fps
                )
            except KeyError:
                # Validated against the static ceiling but fully
                # budget-dropped since: refuse with the compared figures
                # instead of 500ing on the missing cell.
                if job_model_id is None:
                    raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
                min_seconds, max_seconds = self._static_auto_range_or_reject(
                    job_filter,
                    model_id=job_model_id,
                    pipeline=req.model,
                    resolution=resolution,
                    aspect_ratio=req.aspectRatio,
                    fps=fps,
                    allow_evict_retry=True,
                    allow_risk=risk_allowance,
                    unmeasured_bet=unmeasured_bet,
                )
                # The static probe may have evicted the resident pipeline to
                # admit the range; re-sample so the gate below prices the
                # freed card instead of the stale pre-evict filter.
                job_filter = self._local_job_filter(model_id=job_model_id)
            num_frames: int | AutoDurationSpec = AutoDurationSpec(
                min_seconds=float(min_seconds),
                max_seconds=float(max_seconds),
            )
        else:
            num_frames = self._compute_num_frames(duration, fps)

        image = None
        last_image = None
        keyframe_images: list[tuple[Image.Image, int, float]] | None = None
        image_path = normalize_optional_path(req.imagePath)
        last_image_path = normalize_optional_path(req.lastImagePath)
        if req.keyframes:
            keyframe_images = [
                (self._prepare_image(keyframe.imagePath, width, height), keyframe.frameIndex, keyframe.strength)
                for keyframe in req.keyframes
            ]
            logger.info("Keyframes: %s", [(keyframe.imagePath, keyframe.frameIndex) for keyframe in req.keyframes])
            opening = min(req.keyframes, key=lambda keyframe: keyframe.frameIndex)
            image_path = normalize_optional_path(opening.imagePath)
            last_image_path = None
            enhance_keyframes = sorted(
                (
                    (
                        normalize_optional_path(keyframe.imagePath) or keyframe.imagePath,
                        keyframe.frameIndex,
                        keyframe.strength,
                    )
                    for keyframe in req.keyframes
                ),
                key=lambda item: item[1],
            )
        else:
            enhance_keyframes = None
            if image_path:
                image = self._prepare_image(image_path, width, height)
                logger.info("Image: %s -> %sx%s", image_path, width, height)
            if last_image_path:
                last_image = self._prepare_image(last_image_path, width, height)
                logger.info("Last image: %s -> %sx%s", last_image_path, width, height)

        seed = req.seed if req.seed is not None else self._resolve_seed()
        loras = self._resolve_loras(req.loras, local_model_id=job_model_id)

        if job_model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        canvas = local_canvas(resolution, mode="video")
        budget_width, budget_height = budget_shape(
            local_caps(job_model_id), resolution, req.aspectRatio, mode="video"
        )
        load_mode, chunk_pixel_frames = self._plan_video_job(
            budget_width,
            budget_height,
            self._budget_frame_count(num_frames, fps),
            job_filter=job_filter,
            allow_evict_retry=True,
            allow_risk=risk_allowance,
            unmeasured_bet=unmeasured_bet,
        )

        try:
            # Before the pipeline loads and before the generation is marked running: local
            # enhancement needs the VRAM a resident pipeline holds, and evicting a pipeline is
            # refused once a generation is running. Budget 422 already happened so Gemma is
            # not loaded for a job this machine cannot run.
            resolved_prompt = self._prompt_enhancement.resolve_for_generation(
                req.prompt,
                provenance=req.promptProvenance,
                generation_seed=seed,
                explicit_generation_seed=req.seed is not None,
                image_path=image_path,
                last_image_path=last_image_path,
                keyframes=enhance_keyframes,
                duration=req.duration,
                fps=req.fps,
                lora_catalog_ids=_lora_catalog_ids(req.loras),
                local_model_id=job_model_id,
                explore_generation=explore_generation,
            )
            # Recipe-scoped: applied after enhancement so the LLM rewrites the user
            # scene, then the fixed scaffold wraps the result. None (and thus
            # a no-op) for the plain text-to-video feature.
            #
            # skip_recipe_wrap is owned by resolve_for_generation: auto-off + a typed
            # prompt skips the wrap so Generate sends the scene as typed. A manual
            # Enhance (provenance "enhanced") still wraps so the style lock lands on
            # the rewrite the user approved.
            generation_prompt = resolved_prompt.prompt
            enhance_via_api = resolved_prompt.enhance_via_api
            if prompt_wrap is not None and not resolved_prompt.skip_recipe_wrap:
                generation_prompt = prompt_wrap(generation_prompt)
                # The wrap must be the last text mutation: on the API-encoding path
                # resolved_prompt.prompt is the *raw* user scene with
                # enhance_via_api=True, which would let /prompt-embedding rewrite the
                # scaffold we just prepended. Encode the wrapped prompt as-is instead
                # so the style lock survives. Home auto-enhance rewrites locally
                # before this wrap when the enhancer checkpoint is installed.
                if (
                    resolved_prompt.enhance_via_api
                    and not resolved_prompt.enhanced_locally
                    and req.promptProvenance != "enhanced"
                ):
                    logger.info(
                        "Style recipe sent the typed scene: no local prompt enhancer "
                        "is installed, and API enhance would rewrite the scaffold"
                    )
                enhance_via_api = False
            logger.debug("Generation prompt for %s: %s", generation_id, generation_prompt)
            self._generation.raise_if_cancelled()
            self._pipelines.load_gpu_pipeline(
                "fast", loras=loras, mode=load_mode, ltx_model_id=job_model_id
            )
            has_image_input = (
                image is not None or last_image is not None or bool(keyframe_images)
            )
            self._generation.start_generation(
                generation_id,
                estimated=self._estimate_wall_time(
                    width,
                    height,
                    self._budget_frame_count(num_frames, fps),
                    load_mode,
                    "i2v" if has_image_input else "t2v",
                ),
            )

            reference_images = self._resolve_reference_images(req)
            if len(reference_images) > 1:
                if self.config.local_generations_mode != "mac_mlx_q4":
                    raise HTTPError(400, "LOCAL_REFERENCE_I2V_REQUIRES_MAC_MLX_Q4")
                if not isinstance(num_frames, int):
                    raise HTTPError(422, "Multi-reference MLX generation requires an explicit duration")
                written_path = self._generate_local_reference_i2v(
                    prompt=generation_prompt, reference_images=reference_images,
                    width=width, height=height, num_frames=num_frames,
                    fps=fps, seed=seed, camera_motion=req.cameraMotion,
                )
                self._generation.complete_generation(written_path)
                return GenerateVideoCompleteResponse(status="complete", video_path=written_path)

            written_path = self.generate_video(
                prompt=generation_prompt,
                enhance_via_api=enhance_via_api,
                image=image,
                last_image=last_image,
                keyframe_images=keyframe_images,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
                camera_motion=req.cameraMotion,
                negative_prompt=req.negativePrompt,
                loras=loras,
                output_path=output_path,
                load_mode=load_mode,
                ltx_model_id=job_model_id,
                skip_stage_2=canvas.skip_stage_2,
                chunk_pixel_frames=chunk_pixel_frames,
            )

            self._generation.complete_generation(written_path)
            return self._complete_response(written_path, load_mode)

        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if is_cancel_exception(e):
                logger.info("Generation cancelled by user")
                return self._cancelled_response(load_mode)

            raise HTTPError(500, str(e)) from e

    def _resolve_loras(
        self,
        loras: list[LoraEntry],
        *,
        local_model_id: LTXLocalModelId | None = None,
    ) -> list[tuple[str, float]]:
        if loras:
            model_id = local_model_id or self._active_ltx_model_id()
            if model_id is None:
                raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
            if not supports(local_caps(model_id), "user_loras"):
                raise HTTPError(
                    409,
                    "User LoRAs are not supported for the selected offering.",
                    code="UNSUPPORTED_USER_LORAS",
                )
        try:
            return [(str(resolve_lora_ref(self.models_dir, e.ref)), e.scale) for e in loras]
        except ValueError as exc:
            raise HTTPError(400, str(exc)) from exc

    def generate_video(
        self,
        prompt: str,
        enhance_via_api: bool,
        image: Image.Image | None,
        height: int,
        width: int,
        num_frames: int | AutoDurationSpec,
        fps: float,
        seed: int,
        camera_motion: VideoCameraMotion,
        negative_prompt: str,
        loras: list[tuple[str, float]] | None = None,
        last_image: Image.Image | None = None,
        keyframe_images: list[tuple[Image.Image, int, float]] | None = None,
        output_path: Path | None = None,
        *,
        load_mode: LocalGenerationMode,
        ltx_model_id: LTXLocalModelId | None = None,
        skip_stage_2: bool = False,
        chunk_pixel_frames: int | None = None,
    ) -> str:
        t_total_start = time.perf_counter()
        gen_mode = "keyframes" if keyframe_images else "i2v" if image is not None else "t2v"
        frames_log = (
            f"auto {num_frames.min_seconds:g}-{num_frames.max_seconds:g}s"
            if isinstance(num_frames, AutoDurationSpec)
            else f"{num_frames} frames"
        )
        logger.info("[%s] Generation started (model=fast, %dx%d, %s, %d fps)", gen_mode, width, height, frames_log, int(fps))

        self._generation.raise_if_cancelled()

        if keyframe_images:
            images, temp_image_paths = self._keyframe_conditionings(keyframe_images)
        else:
            images, temp_image_paths = self._image_conditionings(
                first=image, last=last_image, num_frames=num_frames
            )

        if output_path is None:
            output_path = self._make_output_path()

        # Appended after any rewrite the caller already applied, so the enhancer can't
        # paraphrase the camera directive away.
        enhanced_prompt = prompt + self.config.camera_motion_prompts.get(camera_motion, "")

        try:
            self._generation.update_progress("loading_model", 5)
            t_load_start = time.perf_counter()
            pipeline_state = self._pipelines.load_gpu_pipeline(
                "fast", loras=loras, mode=load_mode, ltx_model_id=ltx_model_id
            )
            t_load_end = time.perf_counter()
            logger.info("[%s] Pipeline load: %.2fs", gen_mode, t_load_end - t_load_start)

            self._generation.update_progress("encoding_text", 10)
            encoding_method = (
                "api" if not self._text.should_use_local_encoding(ltx_model_id) else "local"
            )
            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(
                enhanced_prompt, enhance_prompt=enhance_via_api, model_id=ltx_model_id
            )
            t_text_end = time.perf_counter()
            logger.info("[%s] Text encoding (%s): %.2fs", gen_mode, encoding_method, t_text_end - t_text_start)

            self._generation.raise_if_cancelled()
            self._generation.update_progress("inference", 15)

            # Guard for the /64 two-stage grid. Half-way values round up: Python's round() is
            # half-to-even, which turned a 544 height into 512 and silently shipped a frame 32px
            # shorter (and off its stated aspect ratio) rather than the nearest legal size.
            height = snap_up_to_multiple(height, 64)
            width = snap_up_to_multiple(width, 64)

            t_inference_start = time.perf_counter()
            with log_heartbeat(f"{gen_mode} inference"):
                pipeline_state.pipeline.generate(
                    prompt=enhanced_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=fps,
                    images=images,
                    output_path=str(output_path),
                    speed_mode=self.state.app_settings.local_mlx_speed_mode if self.config.local_generations_mode == "mac_mlx_q4" and not images else "quality",
                    guide_all_images=bool(keyframe_images),
                    skip_stage_2=skip_stage_2,
                    chunk_pixel_frames=chunk_pixel_frames,
                )
            t_inference_end = time.perf_counter()
            logger.info("[%s] Inference: %.2fs", gen_mode, t_inference_end - t_inference_start)

            # Denoiser interrupt cannot abort VAE decode / ffmpeg; a Stop after the last
            # denoise step still finishes encode, then this check drops the file.
            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise GenerationCancelledError()

            t_total_end = time.perf_counter()
            logger.info("[%s] Total generation: %.2fs (load=%.2fs, text=%.2fs, inference=%.2fs)",
                        gen_mode, t_total_end - t_total_start,
                        t_load_end - t_load_start, t_text_end - t_text_start, t_inference_end - t_inference_start)

            self._generation.update_progress("complete", 100)
            return str(output_path)
        finally:
            self._text.clear_api_embeddings()
            self._unlink_temp_paths(temp_image_paths)

    def _generate_a2v(
        self,
        req: GenerateVideoRequest,
        duration: int,
        fps: int,
        *,
        audio_path: str,
        generation_id: str,
        output_path: Path | None = None,
        num_frames: int | None = None,
        audio_duration_seconds: float | None = None,
        local_model_id: LTXLocalModelId | None = None,
        explore_generation: bool = False,
        risk_allowance: bool | None = None,
        unmeasured_bet: bool = False,
    ) -> GenerateVideoResponse:
        model_id = local_model_id
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        if not supports(local_caps(model_id), "a2v"):
            raise HTTPError(
                409,
                "Audio-to-video is not supported for the selected offering.",
                code="UNSUPPORTED_A2V",
            )
        validated_audio_path = validate_audio_file(audio_path)
        audio_path_str = str(validated_audio_path)

        width, height = self._local_pixels(
            req.resolution,
            req.aspectRatio,
            invalid_code="INVALID_LOCAL_A2V_RESOLUTION",
            local_model_id=model_id,
            mode="a2v",
        )
        width = snap_up_to_multiple(width, 64)
        height = snap_up_to_multiple(height, 64)

        if num_frames is None:
            num_frames = self._compute_num_frames(duration, fps)
        if audio_duration_seconds is not None:
            heard = self._prefix_a2v_audio_seconds(
                model_id=model_id,
                resolution=req.resolution,
                fps=req.fps,
                audio_duration_seconds=audio_duration_seconds,
            )
            if heard is None:
                raise HTTPError(
                    422,
                    "Audio is too long for audio-to-video.",
                    code="INVALID_VIDEO_GENERATION_SPEC",
                )
            audio_duration_seconds = heard

        image = None
        image_path = normalize_optional_path(req.imagePath)
        if image_path:
            image = self._prepare_image(image_path, width, height)

        last_image = None
        last_image_path = normalize_optional_path(req.lastImagePath)
        if last_image_path:
            last_image = self._prepare_image(last_image_path, width, height)

        seed = req.seed if req.seed is not None else self._resolve_seed()
        loras = self._resolve_loras(req.loras, local_model_id=model_id)

        temp_image_paths: list[str] = []

        # Decided pre-try like the Fast path: a 422 here propagates without
        # failing a generation that never started, and the cancelled branch
        # below always sees a bound mode.
        load_mode, chunk_pixel_frames = self._plan_video_job(
            width,
            height,
            num_frames,
            job_filter=self._local_job_filter(model_id=model_id),
            allow_evict_retry=True,
            allow_risk=risk_allowance,
            unmeasured_bet=unmeasured_bet,
        )

        try:
            neg = req.negativePrompt if req.negativePrompt else self.config.default_negative_prompt

            images, temp_image_paths = self._image_conditionings(
                first=image, last=last_image, num_frames=num_frames
            )

            # Same ordering as Fast: budget 422 before Gemma so an over-size cell
            # does not evict a warmed pipeline for a rewrite that will not run.
            resolved_prompt = self._prompt_enhancement.resolve_for_generation(
                req.prompt,
                provenance=req.promptProvenance,
                generation_seed=seed,
                explicit_generation_seed=req.seed is not None,
                image_path=image_path,
                last_image_path=last_image_path,
                lora_catalog_ids=_lora_catalog_ids(req.loras),
                local_model_id=model_id,
                explore_generation=explore_generation,
                audio_path=audio_path_str,
            )
            enhanced_prompt = resolved_prompt.prompt + self.config.camera_motion_prompts.get(req.cameraMotion, "")

            self._generation.raise_if_cancelled()
            a2v_state = self._pipelines.load_a2v_pipeline(
                loras=loras, mode=load_mode, ltx_model_id=model_id
            )
            self._generation.start_generation(
                generation_id,
                estimated=self._estimate_wall_time(
                    width, height, num_frames, load_mode, "a2v"
                ),
            )

            if output_path is None:
                output_path = self._make_output_path()

            self._generation.update_progress("loading_model", 5)
            self._generation.update_progress("encoding_text", 10)
            self._text.prepare_text_encoding(
                enhanced_prompt,
                enhance_prompt=resolved_prompt.enhance_via_api,
                model_id=model_id,
            )
            self._generation.raise_if_cancelled()
            self._generation.update_progress("inference", 15)

            with log_heartbeat("a2v inference"):
                a2v_state.pipeline.generate(
                    prompt=enhanced_prompt,
                    negative_prompt=neg,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=fps,
                    num_inference_steps=distilled_total_steps(),
                    images=images,
                    audio_path=audio_path_str,
                    audio_start_time=0.0,
                    audio_max_duration=(
                        None
                        if audio_duration_seconds is None
                        else max(num_frames / fps, audio_duration_seconds)
                    ),
                    output_path=str(output_path),
                    chunk_pixel_frames=chunk_pixel_frames,
                )

            # Denoiser interrupt cannot abort VAE decode / ffmpeg; a Stop after the last
            # denoise step still finishes encode, then this check drops the file.
            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise GenerationCancelledError()

            self._generation.update_progress("complete", 100)
            self._generation.complete_generation(str(output_path))
            return self._complete_response(str(output_path), load_mode)

        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if is_cancel_exception(e):
                logger.info("Generation cancelled by user")
                return self._cancelled_response(load_mode)
            raise HTTPError(500, str(e)) from e
        finally:
            self._text.clear_api_embeddings()
            self._unlink_temp_paths(temp_image_paths)

    def _image_conditionings(
        self,
        *,
        first: Image.Image | None,
        last: Image.Image | None,
        num_frames: int | AutoDurationSpec,
    ) -> tuple[list[ImageConditioningInput], list[str]]:
        temp_paths: list[str] = []
        images: list[ImageConditioningInput] = []
        if first is not None:
            path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            first.save(path)
            temp_paths.append(path)
            images.append(ImageConditioningInput(path=path, frame_idx=0, strength=1.0))
        if last is not None:
            if first is None:
                raise HTTPError(
                    422,
                    "Last frame requires a first-frame image",
                    code="INVALID_VIDEO_GENERATION_SPEC",
                )
            if not isinstance(num_frames, int):
                raise HTTPError(
                    422,
                    "Last frame cannot be combined with automatic duration",
                    code="INVALID_VIDEO_GENERATION_SPEC",
                )
            path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            last.save(path)
            temp_paths.append(path)
            images.append(ImageConditioningInput(path=path, frame_idx=num_frames - 1, strength=1.0))
        return images, temp_paths

    def _keyframe_conditionings(
        self,
        frames: list[tuple[Image.Image, int, float]],
    ) -> tuple[list[ImageConditioningInput], list[str]]:
        temp_paths: list[str] = []
        images: list[ImageConditioningInput] = []
        for image, frame_idx, strength in frames:
            path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            image.save(path)
            temp_paths.append(path)
            images.append(ImageConditioningInput(path=path, frame_idx=frame_idx, strength=strength))
        return images, temp_paths

    @staticmethod
    def _unlink_temp_paths(paths: list[str]) -> None:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)

    def _prepare_image(self, image_path: str, width: int, height: int) -> Image.Image:
        validated_path = validate_image_file(image_path)
        try:
            img = open_oriented_rgb(validated_path)
        except Exception:
            raise HTTPError(400, f"Invalid image file: {image_path}") from None
        img_w, img_h = img.size
        target_ratio = width / height
        img_ratio = img_w / img_h
        if img_ratio > target_ratio:
            new_h = height
            new_w = int(img_w * (height / img_h))
        else:
            new_w = width
            new_h = int(img_h * (width / img_w))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return resized.crop((left, top, left + width, top + height))

    @staticmethod
    def _make_generation_id() -> str:
        return uuid.uuid4().hex[:8]

    def _budget_frame_count(self, num_frames: int | AutoDurationSpec, fps: int) -> int:
        if isinstance(num_frames, AutoDurationSpec):
            return compute_num_frames(int(num_frames.max_seconds), fps)
        return num_frames

    def _plan_video_job(
        self,
        width: int,
        height: int,
        frames: int,
        *,
        job_filter: LocalJobFilter,
        allow_evict_retry: bool = False,
        allow_risk: bool | None = None,
        unmeasured_bet: bool = False,
    ) -> tuple[LocalGenerationMode, int | None]:
        """Load mode and temporal window (pixel frames, None = one pass) for this job."""
        load, _ = self._video_job_load(
            width,
            height,
            frames,
            job_filter=job_filter,
            allow_evict_retry=allow_evict_retry,
            allow_risk=allow_risk,
            unmeasured_bet=unmeasured_bet,
        )
        return load.mode, load.chunk_pixel_frames

    def _video_job_load_mode(
        self,
        width: int,
        height: int,
        frames: int,
        *,
        job_filter: LocalJobFilter,
        allow_evict_retry: bool = False,
        allow_risk: bool | None = None,
        unmeasured_bet: bool = False,
    ) -> tuple[LocalGenerationMode, GenerationRisk | None]:
        load, risk = self._video_job_load(
            width,
            height,
            frames,
            job_filter=job_filter,
            allow_evict_retry=allow_evict_retry,
            allow_risk=allow_risk,
            unmeasured_bet=unmeasured_bet,
        )
        return load.mode, risk

    def _video_job_load(
        self,
        width: int,
        height: int,
        frames: int,
        *,
        job_filter: LocalJobFilter,
        allow_evict_retry: bool = False,
        allow_risk: bool | None = None,
        unmeasured_bet: bool = False,
    ) -> tuple[VideoJobLoad, GenerationRisk | None]:
        """422 or pick full vs stream (and the window) for this job. Does not change process mode.

        Real-generation paths pass ``allow_evict_retry``: a total-possible CUDA
        reject whose shortfall live weights cover gets one evict-and-resample
        before the 422. Read-only callers (estimate) and total-only preflights
        must leave it off — the former must never mutate GPU state, the latter
        reject total-short by construction. ``allow_risk=None`` follows the
        filter's live mode (every request-scoped caller); executors pass an
        explicit bool parsed from the stored annotation so a mode flip never
        re-gates a queued job. An over bet streams, including when the
        shortfall later grows. An unmeasured bet does not cover a later
        memory miss. Hard rejects raise in every mode.
        """
        decision = job_filter.decide(width, height, frames)
        if isinstance(decision, VideoJobReject) and allow_evict_retry:
            rebuilt = evict_retry_filter(
                feature="fast",
                is_vram_occupancy_miss=(decision.reason == "stream_over"),
                stream_gib=decision.stream_gib,
                seq=decision.seq,
                full_gib=decision.full_gib,
                job_filter=job_filter,
                live_allocated_bytes=self._gpu.get_cuda_live_allocated_bytes(),
                reserved_bytes=self._gpu.get_cuda_reserved_bytes(),
                log_prefix="video",
                evict=self._pipelines.evict_resident_pipeline,
                rebuild=lambda: self._local_job_filter(
                    model_id=job_filter.profile.model_id,
                    total_only=job_filter.budget.total_only,
                ),
            )
            if rebuilt is not None:
                job_filter = rebuilt
                decision = job_filter.decide(width, height, frames)
        window = (
            f" window={decision.chunk_pixel_frames}f"
            if decision.chunk_pixel_frames is not None
            else ""
        )
        if isinstance(decision, VideoJobReject):
            risk = risk_if_allowed(
                risk_for_reject(decision, job_filter),
                job_filter,
                allow_risk,
                unmeasured_bet=unmeasured_bet,
            )
            if risk is not None:
                logger.info(
                    "[video] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> risk:%s over_gib=%s over_ram_gb=%s%s%s",
                    decision.seq,
                    decision.full_gib,
                    decision.stream_gib,
                    risk.tier,
                    risk.over_gib,
                    risk.over_ram_gb,
                    window,
                    vram_snapshot_suffix(
                        allocated_bytes=self._gpu.get_cuda_live_allocated_bytes(),
                        reserved_bytes=self._gpu.get_cuda_reserved_bytes(),
                        free_bytes=job_filter.budget.free_vram_bytes,
                    ),
                )
                return (
                    VideoJobLoad(
                        mode="streaming_models_loading",
                        seq=decision.seq,
                        full_gib=decision.full_gib,
                        stream_gib=decision.stream_gib,
                        chunk_pixel_frames=decision.chunk_pixel_frames,
                    ),
                    risk,
                )
            logger.info(
                "[video] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> reject%s%s",
                decision.seq,
                decision.full_gib,
                decision.stream_gib,
                window,
                vram_snapshot_suffix(
                    allocated_bytes=self._gpu.get_cuda_live_allocated_bytes(),
                    reserved_bytes=self._gpu.get_cuda_reserved_bytes(),
                    free_bytes=job_filter.budget.free_vram_bytes,
                ),
            )
            message, code = reject_message_and_code(decision, job_filter)
            raise HTTPError(422, message, code=code)
        logger.info(
            "[video] job budget seq=%.0f full=%.1fGiB stream=%.1fGiB -> %s%s%s",
            decision.seq,
            decision.full_gib,
            decision.stream_gib,
            decision.mode,
            window,
            vram_snapshot_suffix(
                allocated_bytes=self._gpu.get_cuda_live_allocated_bytes(),
                reserved_bytes=self._gpu.get_cuda_reserved_bytes(),
                free_bytes=job_filter.budget.free_vram_bytes,
            ),
        )
        return decision, unmeasured_load_risk(
            decision, job_filter, feature="fast", allow_risk=allow_risk
        )

    def _load_offload_for_response(
        self, load_mode: LocalGenerationMode | None
    ) -> tuple[VideoLoadMode | None, PipelineOffloadMode | None]:
        """Completion-response modes: the decided load plus the routed offload.

        Offload is derived, not observed — but through the same helper the
        pipeline was constructed with, so derived == actual by construction,
        with no live reads on either side.
        """
        if load_mode is None or load_mode == "unsupported":
            return (None, None)
        return (load_mode, self._offload_mode(load_mode).name)

    def _estimate_wall_time(
        self,
        width: int,
        height: int,
        frames: int,
        load_mode: LocalGenerationMode,
        feature: EtaFeature,
    ) -> WallTimeEstimate | None:
        """ETA v1 for the decided cell. None whenever the table has no answer."""
        if load_mode == "full_models_loading":
            eta_load: EtaLoadMode = "full_models_loading"
        elif load_mode == "streaming_models_loading":
            eta_load = "streaming_models_loading"
        else:
            return None
        _load, offload = self._load_offload_for_response(load_mode)
        if offload is None:
            return None
        return estimate_fast_wall_time_s(
            width=width,
            height=height,
            frames=frames,
            load_mode=eta_load,
            offload_mode=offload,
            feature=feature,
            machine_factor=(
                1.0
                if is_measured_hardware_class(self.config.gpu_device_name)
                else None
            ),
        )

    def _complete_response(
        self, video_path: str, load_mode: LocalGenerationMode | None
    ) -> GenerateVideoCompleteResponse:
        resp_load_mode, resp_offload_mode = self._load_offload_for_response(load_mode)
        return GenerateVideoCompleteResponse(
            status="complete",
            video_path=video_path,
            load_mode=resp_load_mode,
            offload_mode=resp_offload_mode,
        )

    def _cancelled_response(
        self, load_mode: LocalGenerationMode | None
    ) -> GenerateVideoCancelledResponse:
        resp_load_mode, resp_offload_mode = self._load_offload_for_response(load_mode)
        return GenerateVideoCancelledResponse(
            status="cancelled",
            load_mode=resp_load_mode,
            offload_mode=resp_offload_mode,
        )

    def preflight_queued_fast_job(
        self,
        *,
        offering: OfferingId,
        resolution: LTXVideoGenResolution,
        aspect_ratio: LTXQueuedAspectRatio,
        duration: int | None,
        fps: LTXVideoGenFps,
    ) -> GenerationRisk | None:
        """Apply the measured LTX 2.5 Fast policy before a Home/Remote queue insert.

        Raises hard rejects; returns the bet annotation for downgradable
        rejects in reckless mode.
        """
        model_id = resolve_offering_local_model_id(self.models_dir, offering, self.config.local_generations_mode)
        if model_id is None:
            model_id = OFFERING_LOCAL_MODEL_IDS[offering][0]
        # Queued admission gates on total ("will it ever fit"): a running job
        # depresses free, and the queue exists to wait for a free slot.
        job_filter = self._local_job_filter(model_id=model_id, total_only=True)
        if job_filter.budget_mode != "reckless" and not (
            job_filter.profile.is_windows_fast or job_filter.profile.is_mac_fast
        ):
            return
        # Callers resolve auto (i2v start-frame closest-match) before calling;
        # auto reaching here would budget the wrong shape silently.
        assert aspect_ratio != "auto"
        try:
            width, height = budget_shape(
                local_caps(model_id), resolution, aspect_ratio, mode="video"
            )
        except KeyError as exc:
            raise HTTPError(400, "INVALID_LOCAL_RESOLUTION") from exc
        if duration is None:
            frames = self._queued_auto_budget_frames(
                job_filter,
                model_id=model_id,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                fps=fps,
            )
        else:
            frames = self._compute_num_frames(duration, fps)
        return self._video_job_load_mode(width, height, frames, job_filter=job_filter)[1]

    def preflight_queued_a2v_job(
        self,
        *,
        offering: OfferingId,
        resolution: LTXLocalA2VResolution,
        aspect_ratio: LTXQueuedAspectRatio,
        num_frames: int,
    ) -> GenerationRisk | None:
        """Apply the A2V budget before a Home/Remote queue insert.

        Raises hard rejects; returns the bet annotation for downgradable
        rejects in reckless mode. Runs on every route: the frame count was
        derived from the static ceiling, so a duration the filter dropped is
        a budget result here rather than an invalid spec.

        Prices ``budget_shape``, the same canvas as a fully dropped row:
        the 16:9 anchor, or the selected aspect when it is wider. Callers
        resolve auto (start-frame closest-match, 16:9 without one); an
        unresolvable one falls back to the anchor rather than refusing the
        queue insert.
        """
        model_id = resolve_offering_local_model_id(self.models_dir, offering, self.config.local_generations_mode)
        if model_id is None:
            model_id = OFFERING_LOCAL_MODEL_IDS[offering][0]
        # Queued admission gates on total ("will it ever fit"): a running job
        # depresses free, and the queue exists to wait for a free slot.
        job_filter = self._local_job_filter(model_id=model_id, total_only=True)
        aspect: LtxAspectRatio = "16:9" if aspect_ratio == "auto" else aspect_ratio
        try:
            width, height = budget_shape(
                local_caps(model_id), resolution, aspect, mode="a2v"
            )
        except KeyError as exc:
            raise HTTPError(400, "INVALID_LOCAL_A2V_RESOLUTION") from exc
        return self._video_job_load_mode(width, height, num_frames, job_filter=job_filter)[1]

    def _queued_auto_budget_frames(
        self,
        job_filter: LocalJobFilter,
        *,
        model_id: LTXLocalModelId,
        resolution: LTXVideoGenResolution,
        aspect_ratio: LtxAspectRatio,
        fps: LTXVideoGenFps,
    ) -> int:
        item = next(
            (
                candidate
                for candidate in get_local_video_generation_model_specs(
                    model_id,
                    duration_head_ready=self._duration_head_ready_for(model_id),
                    job_filter=job_filter,
                )
                if candidate.pipeline == "fast"
            ),
            None,
        )
        if item is None:
            raise HTTPError(
                422,
                LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
                code=LOCAL_GENERATION_UNSUPPORTED,
            )
        try:
            _min_seconds, max_seconds = supported_duration_range(
                item, resolution=resolution, fps=fps
            )
        except KeyError:
            # Fully budget-dropped resolution/fps: refuse with the compared
            # figures, not the flat "cannot run" line.
            _min_seconds, max_seconds = self._static_auto_range_or_reject(
                job_filter,
                model_id=model_id,
                pipeline="fast",
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                fps=fps,
            )
        return compute_num_frames(max_seconds, fps)

    def _static_auto_range_or_reject(
        self,
        job_filter: LocalJobFilter,
        *,
        model_id: LTXLocalModelId,
        pipeline: str,
        resolution: LTXVideoGenResolution,
        aspect_ratio: LtxAspectRatio,
        fps: LTXVideoGenFps,
        allow_evict_retry: bool = False,
        allow_risk: bool | None = None,
        unmeasured_bet: bool = False,
    ) -> tuple[int, int]:
        """Static (min, max) auto seconds, refusing with the enriched budget 422.

        Fallback when the filtered specs dropped every duration at this
        resolution/fps: the request validated against the static ceiling, so a
        missing cell is a budget refusal, not a bad request. Re-decides the
        max cell on live figures — ``_video_job_load_mode`` raises the
        enriched 422 naming what the job needs vs what the machine has; if it
        loads now (figures moved since list time), the static range stands.
        ``allow_risk=None`` follows the filter's live mode; executors pass the
        stored annotation so a mode flip never re-gates a queued job.
        """
        static_item = next(
            (
                candidate
                for candidate in get_local_video_generation_model_specs(
                    model_id,
                    duration_head_ready=self._duration_head_ready_for(model_id),
                    job_filter=None,
                )
                if candidate.pipeline == pipeline
            ),
            None,
        )
        if static_item is None:
            raise HTTPError(
                422,
                LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
                code=LOCAL_GENERATION_UNSUPPORTED,
            )
        try:
            min_seconds, max_seconds = supported_duration_range(
                static_item, resolution=resolution, fps=fps
            )
        except KeyError as exc:
            raise HTTPError(
                422,
                LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
                code=LOCAL_GENERATION_UNSUPPORTED,
            ) from exc
        budget_width, budget_height = budget_shape(
            local_caps(model_id), resolution, aspect_ratio, mode="video"
        )
        _, _ = self._video_job_load_mode(
            budget_width,
            budget_height,
            compute_num_frames(max_seconds, fps),
            job_filter=job_filter,
            allow_evict_retry=allow_evict_retry,
            allow_risk=allow_risk,
            unmeasured_bet=unmeasured_bet,
        )
        return min_seconds, max_seconds

    def _reject_budget_dropped_a2v_cell(
        self,
        job_filter: LocalJobFilter,
        *,
        resolution: LTXLocalA2VResolution,
        aspect_ratio: LTXQueuedAspectRatio,
        num_frames: int,
        static_cell_frames: int,
    ) -> None:
        """Hard-refuse a resolution the filter dropped entirely.

        The audio-derived frame count is quoted when that shape misses. When
        it loads, ``static_cell_frames`` (the shortest static cell) is quoted
        instead, so the figures match the drop that emptied the row.
        ``allow_risk=False`` so reckless does not annotate a row the picker
        no longer offers. A partial drop does not come here.
        """
        model_id = job_filter.profile.model_id
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        aspect: LtxAspectRatio = "16:9" if aspect_ratio == "auto" else aspect_ratio
        try:
            width, height = budget_shape(
                local_caps(model_id), resolution, aspect, mode="a2v"
            )
        except KeyError as exc:
            raise HTTPError(400, "INVALID_VIDEO_GENERATION_SPEC") from exc
        quoted_frames = num_frames
        if not isinstance(job_filter.decide(width, height, num_frames), VideoJobReject):
            quoted_frames = static_cell_frames
        self._video_job_load_mode(
            width,
            height,
            quoted_frames,
            job_filter=job_filter,
            allow_risk=False,
        )
        raise HTTPError(
            422,
            LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
            code=LOCAL_GENERATION_UNSUPPORTED,
        )

    @staticmethod
    def _compute_num_frames(duration: int, fps: int) -> int:
        return compute_num_frames(duration, fps)

    def _make_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.outputs_dir / f"ltx2_video_{timestamp}_{self._make_generation_id()}.mp4"

    def _generate_forced_api(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        with self._generation.reserved_generation_start():

            generation_id = self._make_generation_id()
            self._generation.start_api_generation(generation_id)

            audio_path = normalize_optional_path(req.audioPath)
            image_path = normalize_optional_path(req.imagePath)
            last_image_path = normalize_optional_path(req.lastImagePath)
            has_input_audio = bool(audio_path)
            has_input_image = bool(image_path)

            try:
                self._generation.update_progress("validating_request", 5, None, None)

                api_key = self.state.app_settings.ltx_api_key.strip()
                logger.info("Forced API generation route selected (key_present=%s)", bool(api_key))
                if not api_key:
                    raise missing_ltx_api_key_error()

                requested_model = req.model
                api_model_id = FORCED_API_MODEL_MAP.get(requested_model)
                if api_model_id is None:
                    raise HTTPError(500, "INVALID_FORCED_API_MODEL_CONFIG")

                resolution_label = req.resolution
                resolution_by_aspect = FORCED_API_RESOLUTION_MAP.get(requested_model, {}).get(resolution_label)
                if resolution_by_aspect is None:
                    raise HTTPError(500, "INVALID_FORCED_API_RESOLUTION_CONFIG")

                aspect_ratio = req.aspectRatio
                api_resolution = resolution_by_aspect.get(aspect_ratio)
                if api_resolution is None:
                    raise HTTPError(400, "INVALID_FORCED_API_ASPECT_RATIO")

                prompt = req.prompt
                enhance_prompt = self._prompt_enhancement.api_enhance_prompt(
                    prompt,
                    provenance=req.promptProvenance,
                    image_path=image_path,
                    audio_path=audio_path,
                )

                self._generation.raise_if_cancelled()

                if has_input_audio:
                    validated_audio_path = validate_audio_file(audio_path)
                    validated_image_path: Path | None = None
                    if image_path is not None:
                        validated_image_path = validate_image_file(image_path)

                    self._generation.update_progress("uploading_audio", 20, None, None)
                    audio_uri = self._ltx_api_client.upload_file(
                        api_key=api_key,
                        file_path=str(validated_audio_path),
                    )
                    image_uri: str | None = None
                    last_frame_uri: str | None = None
                    if validated_image_path is not None:
                        self._generation.update_progress("uploading_image", 35, None, None)
                        image_uri = _upload_oriented_image(
                            self._ltx_api_client,
                            api_key=api_key,
                            path=validated_image_path,
                        )
                    if last_image_path is not None:
                        self._generation.update_progress("uploading_image", 45, None, None)
                        last_frame_uri = _upload_oriented_image(
                            self._ltx_api_client,
                            api_key=api_key,
                            path=validate_image_file(last_image_path),
                        )
                    self._generation.update_progress("inference", 55, None, None)
                    video_bytes = self._ltx_api_client.generate_audio_to_video(
                        api_key=api_key,
                        prompt=prompt,
                        audio_uri=audio_uri,
                        image_uri=image_uri,
                        last_frame_uri=last_frame_uri,
                        model=api_model_id,
                        resolution=api_resolution,
                        enhance_prompt=enhance_prompt,
                    )
                    self._generation.update_progress("downloading_output", 85, None, None)
                elif has_input_image:
                    validated_image_path = validate_image_file(image_path)

                    duration = req.duration
                    fps = req.fps

                    generate_audio = req.audio
                    self._generation.update_progress("uploading_image", 20, None, None)
                    image_uri = _upload_oriented_image(
                        self._ltx_api_client,
                        api_key=api_key,
                        path=validated_image_path,
                    )
                    last_frame_uri: str | None = None
                    if last_image_path is not None:
                        self._generation.update_progress("uploading_image", 35, None, None)
                        last_frame_uri = _upload_oriented_image(
                            self._ltx_api_client,
                            api_key=api_key,
                            path=validate_image_file(last_image_path),
                        )
                    self._generation.update_progress("inference", 55, None, None)
                    video_bytes = self._ltx_api_client.generate_image_to_video(
                        api_key=api_key,
                        prompt=prompt,
                        image_uri=image_uri,
                        last_frame_uri=last_frame_uri,
                        model=api_model_id,
                        resolution=api_resolution,
                        duration=None if duration is None else float(duration),
                        fps=float(fps),
                        generate_audio=generate_audio,
                        camera_motion=req.cameraMotion,
                        enhance_prompt=enhance_prompt,
                    )
                    self._generation.update_progress("downloading_output", 85, None, None)
                else:
                    duration = req.duration
                    fps = req.fps

                    generate_audio = req.audio
                    self._generation.update_progress("inference", 55, None, None)
                    video_bytes = self._ltx_api_client.generate_text_to_video(
                        api_key=api_key,
                        prompt=prompt,
                        model=api_model_id,
                        resolution=api_resolution,
                        duration=None if duration is None else float(duration),
                        fps=float(fps),
                        generate_audio=generate_audio,
                        camera_motion=req.cameraMotion,
                        enhance_prompt=enhance_prompt,
                    )
                    self._generation.update_progress("downloading_output", 85, None, None)

                self._generation.raise_if_cancelled()

                output_path = self._write_forced_api_video(video_bytes)
                if self._generation.is_generation_cancelled():
                    output_path.unlink(missing_ok=True)
                    raise GenerationCancelledError()

                self._generation.update_progress("complete", 100, None, None)
                self._generation.complete_generation(str(output_path))
                return GenerateVideoCompleteResponse(status="complete", video_path=str(output_path))
            except HTTPError as e:
                self._generation.fail_generation(e.detail)
                raise
            except LTXAPIClientError as e:
                mapped_error = map_ltx_api_client_error(e)
                self._generation.fail_generation(mapped_error.detail)
                raise mapped_error from e
            except Exception as e:
                self._generation.fail_generation(str(e))
                if is_cancel_exception(e):
                    logger.info("Generation cancelled by user")
                    return GenerateVideoCancelledResponse(status="cancelled")
                raise HTTPError(500, str(e)) from e

    def _write_forced_api_video(self, video_bytes: bytes) -> Path:
        output_path = self._make_output_path()
        output_path.write_bytes(video_bytes)
        return output_path

    @staticmethod
    def _resolve_reference_images(req: GenerateVideoRequest) -> list[ReferenceImageInput]:
        if req.referenceImages:
            return [
                ReferenceImageInput(path=str(validate_image_file(item.path)), role=item.role)
                for item in req.referenceImages
                if normalize_optional_path(item.path) is not None
            ]
        return [
            ReferenceImageInput(path=str(validate_image_file(path)), role="keyframe")
            for path in req.referenceImagePaths
            if normalize_optional_path(path) is not None
        ]

    def _generate_local_reference_i2v(
        self,
        *,
        prompt: str,
        reference_images: list[ReferenceImageInput],
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
        camera_motion: VideoCameraMotion,
    ) -> str:
        self._generation.update_progress("loading_model", 5, 0, 12)
        try:
            lora_path = get_existing_cp_path(
                self.models_dir,
                "ltx-2.3-22b-ic-lora-union-control-ref0.5",
            )
        except FileNotFoundError as exc:
            raise HTTPError(
                409,
                "Local multi-reference generation requires the IC-LoRA Union Control model. Download it from Settings > Models, then try again.",
                code="LOCAL_REFERENCE_I2V_MODEL_MISSING",
            ) from exc

        prepared_paths: list[str] = []
        control_path: str | None = None
        try:
            prepared_references: list[tuple[Image.Image, ReferenceImageRole]] = [
                (self._prepare_image(item.path, width, height), item.role)
                for item in reference_images
            ]
            for image, _role in prepared_references:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()
                image.save(tmp.name)
                prepared_paths.append(tmp.name)

            self._generation.update_progress("encoding_text", 10, 0, 12)
            pipeline_state = self._pipelines.load_gpu_pipeline("fast")
            enhanced_prompt = prompt + self.config.camera_motion_prompts.get(camera_motion, "")
            self._generation.update_progress("preparing_references", 14, 0, 12)
            control_path = self._make_reference_control_video(
                prepared_references=prepared_references,
                width=width,
                height=height,
                num_frames=self._reference_control_frame_count(num_frames),
                fps=fps,
            )
            keyframe_paths = [
                path
                for path, item in zip(prepared_paths, reference_images, strict=True)
                if item.role == "keyframe"
            ]
            frame_indices = self._reference_frame_indices(len(keyframe_paths), num_frames)
            anchors = [
                ImageConditioningInput(path=path, frame_idx=frame_idx, strength=1.0)
                for path, frame_idx in zip(keyframe_paths, frame_indices, strict=True)
            ]

            output_path = self._make_output_path()
            self._generation.update_progress("inference", 15, 0, 12)
            pipeline_state.pipeline.generate_reference_i2v(
                prompt=enhanced_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                images=anchors,
                video_conditioning=[(control_path, 1.0)],
                lora_path=str(lora_path),
                output_path=str(output_path),
            )
            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise RuntimeError("Generation was cancelled")
            self._generation.update_progress("complete", 100, 12, 12)
            return str(output_path)
        finally:
            for path in prepared_paths:
                if os.path.exists(path):
                    os.unlink(path)
            if control_path and os.path.exists(control_path):
                os.unlink(control_path)

    @staticmethod
    def _reference_frame_indices(reference_count: int, num_frames: int) -> list[int]:
        if reference_count <= 0:
            return []
        if reference_count <= 1:
            return [0]
        last = max(num_frames - 1, 0)
        return [
            round(last * index / (reference_count - 1))
            for index in range(reference_count)
        ]

    @staticmethod
    def _reference_control_frame_count(num_frames: int) -> int:
        return num_frames

    def _make_reference_control_video(
        self,
        *,
        prepared_references: list[tuple[Image.Image, ReferenceImageRole]],
        width: int,
        height: int,
        num_frames: int,
        fps: int,
    ) -> str:
        try:
            import cv2
            import imageio_ffmpeg  # type: ignore[reportMissingTypeStubs]
            import numpy as np

            ffmpeg = str(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception as exc:
            raise HTTPError(500, "Local multi-reference generation requires OpenCV and bundled ffmpeg") from exc

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        output_path = tmp.name
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-frames:v",
            str(num_frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        # A file avoids stderr pipe backpressure while raw frames stream on stdin.
        with tempfile.TemporaryFile() as errors:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            assert proc.stdin is not None
            try:
                total = max(num_frames - 1, 1)
                role_weights: dict[ReferenceImageRole, float] = {
                    "character": 0.85,
                    "location": 0.65,
                    "style": 0.35,
                    "keyframe": 1.0,
                }
                edge_references: list[tuple[Any, ReferenceImageRole]] = []
                for image, role in prepared_references:
                    rgb = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    edge_references.append((edges, role))
                persistent_edges: list[tuple[Any, ReferenceImageRole]] = [
                    (edges, role)
                    for edges, role in edge_references
                    if role != "keyframe"
                ]
                for frame_idx in range(num_frames):
                    if self._generation.is_generation_cancelled():
                        raise RuntimeError("Generation was cancelled")
                    if persistent_edges:
                        combined = np.zeros((height, width), dtype=np.uint8)
                        for edges, role in persistent_edges:
                            weighted = np.clip(
                                edges.astype(np.float32) * role_weights[role],
                                0,
                                255,
                            ).astype(np.uint8)
                            combined = np.maximum(combined, weighted)
                        edges = combined
                    else:
                        ref_index = round((len(edge_references) - 1) * frame_idx / total)
                        edges = edge_references[ref_index][0]
                    edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    proc.stdin.write(edge_rgb.tobytes())
                    if frame_idx % max(fps, 1) == 0:
                        progress = 14 + int(1 * frame_idx / total)
                        self._generation.update_progress("preparing_references", progress, frame_idx, num_frames)
                proc.stdin.close()
                return_code = proc.wait(timeout=120)
                errors.seek(0)
                stderr = errors.read().decode("utf-8", errors="replace")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=3)
                if proc.stdin and not proc.stdin.closed:
                    try:
                        proc.stdin.close()
                    except BrokenPipeError:
                        pass

        if return_code != 0:
            Path(output_path).unlink(missing_ok=True)
            raise HTTPError(500, f"Reference control video creation failed: {stderr.strip()}")
        return output_path
