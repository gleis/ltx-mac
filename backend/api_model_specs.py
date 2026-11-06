"""Canonical video-generation capability specs used by backend and frontend."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from api_types import (
    BudgetReadout,
    ChunkedFastCell,
    DownloadedLocalVideoGenerationModelSpecItem,
    DroppedFastAspectCell,
    DroppedFastCell,
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    LTXLocalModelId,
    LTXOfferingCapabilitiesSpec,
    LTXVideoGenerationModelSpecItem,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    LTXVideoGenAspectRatio,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenPipeline,
    LTXVideoGenResolution,
    LocalUnavailableReason,
    WarnedFastAspectCell,
    WarnedFastCell,
)
from frame_math import compute_num_frames
from keyframe_request import validate_keyframe_inputs
from runtime_config.fast_chunking import job_window_count
from runtime_config.ltx_capabilities import (
    CanvasMode,
    LtxOfferingCapabilities,
    api_caps,
    budget_pixels,
    budget_size,
    effective_local_caps,
    local_canvas,
    pixels_for,
)
from runtime_config.offerings import (
    OFFERING_IDS,
    offering_id_for_local_model_id,
    resolve_offering_local_model_id,
)
from runtime_config.model_download_specs import (
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    is_duration_head_ready,
)
from runtime_config.reject_reasons import (
    LoadRiskTier,
    ShortfallKind,
    VideoJobRejectReason,
)
from runtime_config.video_job_budget import (
    BudgetFeature,
    LocalJobFilter,
    VideoJobLoad,
    VideoJobReject,
    decide_fast_durations,
    decide_video_job,
    load_headroom_gib,
    load_risk_tier,
    reject_message_and_code,
    reject_over_gib,
    reject_over_ram_gb,
    reject_shortfall_kind,
    budget_readout_for_reject,
    risk_for_reject,
    tight_fit_message,
    tight_readout,
    unmeasured_message,
    untested_readout,
)
from server_utils.media_validation import normalize_optional_path

# The concrete ltxv-api model id each Desktop-facing pipeline maps to when forced onto
# the API backend. SSOT for this mapping — video_generation_handler and the
# retake/extend handlers all import it from here.
FORCED_API_MODEL_MAP: dict[str, str] = {
    "fast": "ltx-2-3-fast",
    "pro": "ltx-2-3-pro",
    "fast-2.5": "ltx-2-5-fast",
    "pro-2.5": "ltx-2-5-pro",
}


_ApiDurationEnvelope = tuple[LTXVideoGenDuration, ...]
_ApiFpsDurationMap = dict[LTXVideoGenFps, _ApiDurationEnvelope]
_ApiResolutionMap = dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec]

# API duration envelopes (seconds). ltxv-api accepts 2–20s. GenSpace floors the
# picker at 6s; gap fill uses smallest_valid from this full list.
# 20s is Fast 720p/1080p at 24/25 and the A2V standard-tier audio cap.
_API_DURATIONS_TO_10S: _ApiDurationEnvelope = (2, 3, 4, 5, 6, 8, 10)
_API_DURATIONS_TO_20S: _ApiDurationEnvelope = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20)

_API_FPS_STANDARD: _ApiFpsDurationMap = {
    24: _API_DURATIONS_TO_10S,
    25: _API_DURATIONS_TO_10S,
    48: _API_DURATIONS_TO_10S,
    50: _API_DURATIONS_TO_10S,
}
_API_FPS_EXTENDED: _ApiFpsDurationMap = {
    24: _API_DURATIONS_TO_20S,
    25: _API_DURATIONS_TO_20S,
    48: _API_DURATIONS_TO_10S,
    50: _API_DURATIONS_TO_10S,
}
# ltx-2-5-pro has no 48 fps in MODEL_CAPABILITY_MATRIX.
_API_FPS_PRO_2_5: _ApiFpsDurationMap = {
    24: _API_DURATIONS_TO_10S,
    25: _API_DURATIONS_TO_10S,
    50: _API_DURATIONS_TO_10S,
}


# 2.3 generate is 16:9/9:16. 2.5 adds 1:1 and 3:2. 21:9, 4:3, and 4:5 are local-only.
_API_ASPECT_RATIOS_2_3: tuple[LTXVideoGenAspectRatio, ...] = ("16:9", "9:16")
_API_ASPECT_RATIOS_2_5: tuple[LTXVideoGenAspectRatio, ...] = ("16:9", "3:2", "1:1", "9:16")


def _resolution_spec(
    fps_to_durations: _ApiFpsDurationMap,
    aspect_ratios: tuple[LTXVideoGenAspectRatio, ...],
) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={
            fps: list(durations)
            for fps, durations in fps_to_durations.items()
        },
        aspect_ratios=list(aspect_ratios),
    )


def _resolution_map(
    cells: dict[LTXVideoGenResolution, _ApiFpsDurationMap],
    aspect_ratios: tuple[LTXVideoGenAspectRatio, ...],
) -> _ApiResolutionMap:
    return {
        resolution: _resolution_spec(fps_to_durations, aspect_ratios)
        for resolution, fps_to_durations in cells.items()
    }


# Fast t2v/i2v: 720p/1080p get 20s at 24/25; 1440p/4K stay at 10s.
_API_FAST_CELLS: dict[LTXVideoGenResolution, _ApiFpsDurationMap] = {
    "720p": _API_FPS_EXTENDED,
    "1080p": _API_FPS_EXTENDED,
    "1440p": _API_FPS_STANDARD,
    "2160p": _API_FPS_STANDARD,
}
# Pro 2.3 t2v/i2v: 10s at every fps and resolution, including 720p.
_API_PRO_CELLS: dict[LTXVideoGenResolution, _ApiFpsDurationMap] = {
    "720p": _API_FPS_STANDARD,
    "1080p": _API_FPS_STANDARD,
    "1440p": _API_FPS_STANDARD,
    "2160p": _API_FPS_STANDARD,
}
# Pro 2.5 t2v/i2v: 720p+1080p, 24/25/50, 10s. No 48 fps, no 1440p/4K.
_API_PRO_2_5_CELLS: dict[LTXVideoGenResolution, _ApiFpsDurationMap] = {
    "720p": _API_FPS_PRO_2_5,
    "1080p": _API_FPS_PRO_2_5,
}

_API_FAST_RESOLUTIONS = _resolution_map(_API_FAST_CELLS, _API_ASPECT_RATIOS_2_3)
_API_FAST_25_RESOLUTIONS = _resolution_map(_API_FAST_CELLS, _API_ASPECT_RATIOS_2_5)
_API_PRO_RESOLUTIONS = _resolution_map(_API_PRO_CELLS, _API_ASPECT_RATIOS_2_3)
_API_PRO_2_5_RESOLUTIONS = _resolution_map(_API_PRO_2_5_CELLS, _API_ASPECT_RATIOS_2_5)
# A2V stays 16:9/9:16 even on 2.5. 20s audio at 720p/1080p, 10s at 1440p/4K.
_API_A2V_RESOLUTIONS = _resolution_map(_API_FAST_CELLS, _API_ASPECT_RATIOS_2_3)
_API_PRO_2_5_A2V_RESOLUTIONS = _resolution_map(_API_PRO_2_5_CELLS, _API_ASPECT_RATIOS_2_3)


ltx_api_model_specs: tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...] = (
    (
        "fast",
        LTXVideoGenerationSpec(
            display_name="LTX-2.3 Fast (API)",
            supported_resolutions_durations=_API_FAST_RESOLUTIONS,
            # No A2V envelope: ltxv-api audio-to-video does not accept ltx-2-3-fast.
        ),
    ),
    (
        "pro",
        LTXVideoGenerationSpec(
            display_name="LTX-2.3 Pro (API)",
            supported_resolutions_durations=_API_PRO_RESOLUTIONS,
            a2v_supported_resolutions_durations=_API_A2V_RESOLUTIONS,
        ),
    ),
    (
        "fast-2.5",
        LTXVideoGenerationSpec(
            display_name="LTX-2.5 Fast (API)",
            supported_resolutions_durations=_API_FAST_25_RESOLUTIONS,
            a2v_supported_resolutions_durations=_API_A2V_RESOLUTIONS,
        ),
    ),
    (
        "pro-2.5",
        LTXVideoGenerationSpec(
            display_name="LTX-2.5 Pro (API)",
            supported_resolutions_durations=_API_PRO_2_5_RESOLUTIONS,
            a2v_supported_resolutions_durations=_API_PRO_2_5_A2V_RESOLUTIONS,
        ),
    ),
)


def forced_api_resolution_map() -> dict[str, dict[str, dict[str, str]]]:
    """Pipeline → resolution label → aspect → WxH, taken from each API spec cell."""
    mapped: dict[str, dict[str, dict[str, str]]] = {}
    for pipeline, spec in ltx_api_model_specs:
        caps = api_caps(pipeline)
        by_resolution: dict[str, dict[str, str]] = {}
        for resolution, cell in spec.supported_resolutions_durations.items():
            by_resolution[resolution] = {
                aspect: f"{width}x{height}"
                for aspect in cell.aspect_ratios
                for width, height in (pixels_for(caps, resolution, aspect),)
            }
        mapped[pipeline] = by_resolution
    return mapped


def _capabilities_spec(caps: LtxOfferingCapabilities) -> LTXOfferingCapabilitiesSpec:
    return LTXOfferingCapabilitiesSpec(
        t2v=caps.t2v,
        i2v=caps.i2v,
        a2v=caps.a2v,
        ic_lora=caps.ic_lora,
        retake=caps.retake,
        extend=caps.extend,
        multi_keyframe=caps.multi_keyframe,
        multi_keyframe_max_count=caps.multi_keyframe_max_count,
        user_loras=caps.user_loras,
        camera_motion=caps.camera_motion,
        auto_duration=caps.auto_duration,
    )


def _item_with_caps(
    pipeline: LTXVideoGenPipeline,
    spec: LTXVideoGenerationSpec,
    caps: LtxOfferingCapabilities,
) -> LTXVideoGenerationModelSpecItem:
    return LTXVideoGenerationModelSpecItem(
        pipeline=pipeline,
        spec=spec.model_copy(
            update={
                "capabilities": _capabilities_spec(caps),
                "a2v_supported_resolutions_durations": (
                    spec.a2v_supported_resolutions_durations if caps.a2v else None
                ),
            }
        ),
    )


def _dropped_cell_message(
    decision: VideoJobReject,
    job_filter: LocalJobFilter,
) -> str:
    """List-time refusal copy for one withheld cell.

    Same builder as the request-time 422s over the figures the gate
    compared, so a cell refused in the picker reads identically if it
    ever 422s.
    """
    message, _code = reject_message_and_code(decision, job_filter)
    return message


_UNSUPPORTED_DROPPED_DECISION = VideoJobReject(
    seq=0.0, full_gib=0.0, stream_gib=0.0, reason="unsupported"
)


def _unsupported_dropped_cells(
    specs: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec],
    *,
    audio: bool,
    job_filter: LocalJobFilter,
) -> list[DroppedFastCell]:
    """Every static cell withheld with `unsupported`, no decisions needed."""
    return [
        DroppedFastCell(
            resolution=resolution,
            fps=fps,
            duration=duration,
            reason="unsupported",
            message=_dropped_cell_message(
                _UNSUPPORTED_DROPPED_DECISION, job_filter
            ),
            audio=audio,
        )
        for resolution, spec in specs.items()
        for fps, durations in spec.fps_to_durations.items()
        for duration in durations
    ]


def _warned_cell_parts(
    load: VideoJobLoad,
    job_filter: LocalJobFilter,
    *,
    feature: BudgetFeature,
) -> tuple[LoadRiskTier, float | None, str, BudgetReadout] | None:
    """Warning tier, headroom, and tooltip for a kept cell, if it warns.

    Single home for warned-cell emission (base cells and aspect cells): tight
    cells carry available-minus-ask headroom, unmeasured cells compare nothing.
    None when the cell fits cleanly.
    """
    tier = load_risk_tier(load, job_filter, feature=feature)
    if tier is None:
        return None
    if tier == "unmeasured":
        return (tier, None, unmeasured_message(), untested_readout())
    headroom = load_headroom_gib(load, job_filter)
    assert headroom is not None  # a tight tier implies compared figures exist
    available = headroom + load.stream_gib
    darwin = job_filter.profile.system == "Darwin"
    return (
        tier,
        headroom,
        tight_fit_message(load.stream_gib, available, darwin=darwin),
        tight_readout(load.stream_gib, available),
    )


def _dropped_cell_figures(
    reject: VideoJobReject, job_filter: LocalJobFilter
) -> tuple[float | None, int | None, ShortfallKind | None]:
    """Figure triple for one withheld cell: one ``risk_for_reject`` call, unpacked.

    When it declines (hard wall, or no compared exact bytes) the per-figure
    helpers still name an integer-fallback total binding, so the fallback keeps
    those rows byte-identical.
    """
    risk = risk_for_reject(reject, job_filter)
    if risk is not None:
        return (risk.over_gib, risk.over_ram_gb, risk.shortfall_kind)
    return (
        reject_over_gib(reject, job_filter),
        reject_over_ram_gb(reject, job_filter),
        reject_shortfall_kind(reject, job_filter),
    )


def _filter_local_fast_resolution_map(
    specs: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None,
    caps: LtxOfferingCapabilities,
    *,
    job_filter: LocalJobFilter,
    mode: CanvasMode = "video",
    audio: bool = False,
) -> tuple[
    dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None,
    list[DroppedFastCell],
    list[DroppedFastAspectCell],
    list[WarnedFastCell],
    list[WarnedFastAspectCell],
    list[ChunkedFastCell],
]:
    if specs is None:
        return None, [], [], [], [], []
    if job_filter.profile.process_mode == "unsupported":
        return {}, _unsupported_dropped_cells(specs, audio=audio, job_filter=job_filter), [], [], [], []
    filtered: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] = {}
    dropped: list[DroppedFastCell] = []
    aspect_dropped: list[DroppedFastAspectCell] = []
    warned: list[WarnedFastCell] = []
    aspect_warned: list[WarnedFastAspectCell] = []
    chunked: list[ChunkedFastCell] = []
    for resolution, spec in specs.items():
        if resolution not in caps.resolution_pixels_16_9:
            raise KeyError(resolution)
        width, height = budget_size(local_canvas(resolution, mode=mode))
        fps_to_durations: dict[LTXVideoGenFps, list[LTXVideoGenDuration]] = {}
        for fps, durations in spec.fps_to_durations.items():
            outcome = decide_fast_durations(
                width,
                height,
                int(fps),
                durations,
                job_filter=job_filter,
            )
            if outcome.kept:
                fps_to_durations[fps] = cast(
                    list[LTXVideoGenDuration], outcome.kept
                )
            for duration, reason in outcome.rejected.items():
                reject = outcome.rejects[duration]
                over_gib, over_ram_gb, shortfall_kind = _dropped_cell_figures(
                    reject, job_filter
                )
                reject_window = reject.chunk_pixel_frames
                dropped.append(
                    DroppedFastCell(
                        resolution=resolution,
                        fps=fps,
                        duration=cast(LTXVideoGenDuration, duration),
                        reason=reason,
                        message=_dropped_cell_message(reject, job_filter),
                        audio=audio,
                        over_gib=over_gib,
                        over_ram_gb=over_ram_gb,
                        shortfall_kind=shortfall_kind,
                        readout=budget_readout_for_reject(reject, job_filter),
                        windows=(
                            None
                            if reject_window is None
                            else job_window_count(
                                compute_num_frames(duration, int(fps)), reject_window
                            )
                        ),
                        window_frames=reject_window,
                    )
                )
            # Emit warned cells for kept durations
            for duration in outcome.kept:
                parts = _warned_cell_parts(
                    outcome.loads[duration], job_filter, feature="fast"
                )
                if parts is None:
                    continue
                tier, headroom_gib, message, readout = parts
                warned.append(
                    WarnedFastCell(
                        resolution=resolution,
                        fps=fps,
                        duration=cast(LTXVideoGenDuration, duration),
                        tier=tier,
                        headroom_gib=headroom_gib,
                        message=message,
                        readout=readout,
                        audio=audio,
                    )
                )
            # Kept durations that run in more than one window, for the "takes longer" notice.
            for duration in outcome.kept:
                window = outcome.loads[duration].chunk_pixel_frames
                if window is None:
                    continue
                chunked.append(
                    ChunkedFastCell(
                        resolution=resolution,
                        fps=fps,
                        duration=cast(LTXVideoGenDuration, duration),
                        windows=job_window_count(compute_num_frames(duration, int(fps)), window),
                        window_frames=window,
                        audio=audio,
                    )
                )
            # Aspect refinement: a kept anchor cell a wider shape cannot run.
            # Narrower-or-equal shapes cost no more than the anchor, so only
            # wider ratios (21:9 today) can emit here; the anchor itself and
            # anchor-dropped cells never do.
            for duration in outcome.kept:
                frames = compute_num_frames(duration, int(fps))
                for ratio in spec.aspect_ratios:
                    selected_width, selected_height = budget_pixels(
                        caps, resolution, ratio, mode=mode
                    )
                    if selected_width * selected_height <= width * height:
                        continue
                    decision = decide_video_job(
                        selected_width, selected_height, frames, job_filter=job_filter
                    )
                    if isinstance(decision, VideoJobReject):
                        over_gib, over_ram_gb, shortfall_kind = _dropped_cell_figures(
                            decision, job_filter
                        )
                        aspect_dropped.append(
                            DroppedFastAspectCell(
                                resolution=resolution,
                                fps=fps,
                                duration=cast(LTXVideoGenDuration, duration),
                                aspectRatio=ratio,
                                reason=decision.reason,
                                message=_dropped_cell_message(decision, job_filter),
                                audio=audio,
                                over_gib=over_gib,
                                over_ram_gb=over_ram_gb,
                                shortfall_kind=shortfall_kind,
                                readout=budget_readout_for_reject(decision, job_filter),
                            )
                        )
                    else:
                        parts = _warned_cell_parts(decision, job_filter, feature="fast")
                        if parts is not None:
                            tier, headroom_gib, message, readout = parts
                            aspect_warned.append(
                                WarnedFastAspectCell(
                                    resolution=resolution,
                                    fps=fps,
                                    duration=cast(LTXVideoGenDuration, duration),
                                    aspectRatio=ratio,
                                    tier=tier,
                                    headroom_gib=headroom_gib,
                                    message=message,
                                    readout=readout,
                                    audio=audio,
                                )
                            )
        if fps_to_durations:
            filtered[resolution] = LTXVideoGenerationResolutionSpec(
                fps_to_durations=fps_to_durations,
                aspect_ratios=list(spec.aspect_ratios),
            )
    return filtered, dropped, aspect_dropped, warned, aspect_warned, chunked


def _filter_local_fast_item(
    item: LTXVideoGenerationModelSpecItem,
    caps: LtxOfferingCapabilities,
    *,
    job_filter: LocalJobFilter,
) -> LTXVideoGenerationModelSpecItem:
    spec = item.spec
    kept, dropped, aspect_dropped, warned, aspect_warned, chunked = _filter_local_fast_resolution_map(
        spec.supported_resolutions_durations,
        caps,
        job_filter=job_filter,
    )
    a2v_kept, a2v_dropped, a2v_aspect_dropped, a2v_warned, a2v_aspect_warned, a2v_chunked = _filter_local_fast_resolution_map(
        spec.a2v_supported_resolutions_durations,
        caps,
        job_filter=job_filter,
        mode="a2v",
        audio=True,
    )
    return item.model_copy(
        update={
            "spec": spec.model_copy(
                update={
                    "supported_resolutions_durations": kept,
                    "a2v_supported_resolutions_durations": a2v_kept,
                    "dropped_cells": dropped + a2v_dropped,
                    "dropped_aspect_cells": aspect_dropped + a2v_aspect_dropped,
                    "warned_cells": warned + a2v_warned,
                    "warned_aspect_cells": aspect_warned + a2v_aspect_warned,
                    "chunked_cells": chunked + a2v_chunked,
                }
            )
        }
    )


def get_local_video_generation_model_specs(
    model_id: LTXLocalModelId | None = None,
    *,
    duration_head_ready: bool = False,
    job_filter: LocalJobFilter | None = None,
) -> list[LTXVideoGenerationModelSpecItem]:
    """Specs for one local model, filtered when a job filter is provided.

    ``job_filter=None`` serves the unfiltered static ceiling (request-shape
    validation only). Otherwise each row is evaluated under its own model's
    profile: a downloaded 2.5 row takes the 2.5 Windows profile even while
    2.3 is active, because profile is (system, model, hardware), not offering.
    """
    if job_filter is not None and model_id is None and job_filter.profile.model_id is not None:
        model_id = job_filter.profile.model_id
    resolved_id = model_id or get_latest_ltx_model_id()
    local_model_spec = get_ltx_model_spec(resolved_id)
    caps = effective_local_caps(resolved_id, duration_head_ready=duration_head_ready)
    items = [
        _item_with_caps(pipeline, spec, caps)
        for pipeline, spec in local_model_spec.supported_pipelines
    ]
    if job_filter is None:
        return items
    row_filter = replace(
        job_filter, profile=replace(job_filter.profile, model_id=resolved_id)
    )
    return [_filter_local_fast_item(item, caps, job_filter=row_filter) for item in items]


def get_api_video_generation_model_specs() -> list[LTXVideoGenerationModelSpecItem]:
    # API Auto duration is a cloud capability. Do not gate it on the local DurationHead file.
    return [
        _item_with_caps(pipeline, spec, api_caps(pipeline))
        for pipeline, spec in ltx_api_model_specs
    ]


def get_downloaded_local_video_generation_model_specs(
    models_dir: Path,
    *,
    job_filter: LocalJobFilter,
) -> list[DownloadedLocalVideoGenerationModelSpecItem]:
    items: list[DownloadedLocalVideoGenerationModelSpecItem] = []
    for offering in OFFERING_IDS:
        model_id = resolve_offering_local_model_id(models_dir, offering, "mac_mlx_q4" if job_filter.profile.system == "Darwin" else None)
        if model_id is None:
            continue
        for local_item in get_local_video_generation_model_specs(
            model_id,
            duration_head_ready=is_duration_head_ready(models_dir, model_id),
            job_filter=job_filter,
        ):
            items.append(
                DownloadedLocalVideoGenerationModelSpecItem(
                    model=offering,
                    pipeline=local_item.pipeline,
                    spec=local_item.spec,
                )
            )
    return items


def _fast_grid_reject_reasons(
    specs: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None,
    caps: LtxOfferingCapabilities,
    *,
    job_filter: LocalJobFilter,
) -> set[VideoJobRejectReason]:
    """Every reject reason the grid produces under the served filter.

    Mirrors ``_filter_local_fast_resolution_map``'s grid walk (same cells,
    same filter) but aggregates verdicts instead of maps — no counterfactual
    inputs. A second reason later is a set-membership check, not a new probe.
    Single-pass collection would thread outcomes through four pydantic-returning
    layers to save milliseconds of pure math on an infrequent endpoint — the
    mirror stays.
    """
    reasons: set[VideoJobRejectReason] = set()
    if specs is None:
        return reasons
    for resolution, spec in specs.items():
        if resolution not in caps.resolution_pixels_16_9:
            raise KeyError(resolution)
        width, height = budget_size(local_canvas(resolution, mode="video"))
        for fps, durations in spec.fps_to_durations.items():
            outcome = decide_fast_durations(
                width,
                height,
                int(fps),
                durations,
                job_filter=job_filter,
            )
            reasons.update(outcome.rejected.values())
    return reasons


@dataclass(frozen=True, slots=True)
class _LocalUnavailableDetails:
    """Why the local grid is empty, plus RAM numbers when RAM binds.

    Numbers ride with the reason from the same verdicts: required is the floor
    that actually gated this decision (read from the profile that produced it,
    not a module constant), detected is the measured total RAM. Both None
    unless the reason is ram_shortfall.
    """

    reason: LocalUnavailableReason | None
    required_ram_gb: int | None
    detected_ram_gb: int | None


_EMPTY_DETAILS = _LocalUnavailableDetails(reason=None, required_ram_gb=None, detected_ram_gb=None)


def _local_unavailable_details(
    local_models: list[LTXVideoGenerationModelSpecItem],
    *,
    local_model_id: LTXLocalModelId | None,
    duration_head_ready: bool,
    job_filter: LocalJobFilter,
) -> _LocalUnavailableDetails:
    """Name RAM when it is the binding constraint on an empty local grid.

    A ``ram_shortfall`` verdict means VRAM and mode already passed, so any
    such verdict under the served filter proves RAM is what hides the grid —
    no sufficient-RAM rerun needed. Unknown RAM yields ``unsupported``, never
    shortfall, so it correctly reports empty details here.
    """
    fast = next((item for item in local_models if item.pipeline == "fast"), None)
    if fast is None or fast.spec.supported_resolutions_durations:
        return _EMPTY_DETAILS
    # Same row resolution as get_local_video_generation_model_specs: the probe
    # must evaluate the model the served maps came from.
    resolved_id = local_model_id or job_filter.profile.model_id or get_latest_ltx_model_id()
    row_filter = replace(job_filter, profile=replace(job_filter.profile, model_id=resolved_id))
    fast_spec = next(
        (
            spec
            for pipeline, spec in get_ltx_model_spec(resolved_id).supported_pipelines
            if pipeline == "fast"
        ),
        None,
    )
    if fast_spec is None:
        return _EMPTY_DETAILS
    caps = effective_local_caps(resolved_id, duration_head_ready=duration_head_ready)
    reasons = _fast_grid_reject_reasons(
        fast_spec.supported_resolutions_durations, caps, job_filter=row_filter
    )
    if job_filter.budget_mode == "reckless" and not reasons.isdisjoint(
        {"stream_over", "ram_shortfall"}
    ):
        # Downgradable verdicts exist: reckless renders the warned form, not
        # the empty-state card, so no reason may survive the mode flip.
        return _EMPTY_DETAILS
    if "ram_shortfall" not in reasons:
        return _EMPTY_DETAILS
    # Shortfall verdicts prove total RAM is known (unknown yields unsupported),
    # so detected is present — the assert narrows what the verdicts guarantee.
    total_ram_bytes = row_filter.budget.total_ram_bytes
    assert total_ram_bytes is not None
    return _LocalUnavailableDetails(
        reason="ram_shortfall",
        required_ram_gb=row_filter.profile.windows_fast_required_ram_bytes // (1024**3),
        detected_ram_gb=total_ram_bytes // (1024**3),
    )


def build_generate_video_model_specs_response(
    local_model_id: LTXLocalModelId | None = None,
    *,
    models_dir: Path | None = None,
    duration_head_ready: bool = False,
    job_filter: LocalJobFilter,
    low_performance_machine: bool = False,
) -> GenerateVideoModelsSpecsResponse:
    if local_model_id is None and job_filter.profile.model_id is not None:
        local_model_id = job_filter.profile.model_id
    downloaded = (
        get_downloaded_local_video_generation_model_specs(
            models_dir,
            job_filter=job_filter,
        )
        if models_dir is not None
        else []
    )
    local_models = get_local_video_generation_model_specs(
        local_model_id,
        duration_head_ready=duration_head_ready,
        job_filter=job_filter,
    )
    unavailable = _local_unavailable_details(
        local_models,
        local_model_id=local_model_id,
        duration_head_ready=duration_head_ready,
        job_filter=job_filter,
    )
    return GenerateVideoModelsSpecsResponse(
        local_models=local_models,
        api_models=get_api_video_generation_model_specs(),
        downloaded_local_models=downloaded,
        active_offering=(
            offering_id_for_local_model_id(local_model_id)
            if local_model_id is not None
            else None
        ),
        local_unavailable_reason=unavailable.reason,
        local_unavailable_required_ram_gb=unavailable.required_ram_gb,
        local_unavailable_detected_ram_gb=unavailable.detected_ram_gb,
        low_performance_machine=low_performance_machine,
        budgetMode=job_filter.budget_mode,
    )


def _get_resolution_spec(
    item: LTXVideoGenerationModelSpecItem,
    *,
    resolution: LTXVideoGenResolution,
    is_a2v: bool,
) -> LTXVideoGenerationResolutionSpec | None:
    if is_a2v:
        # A pipeline with no a2v spec doesn't support audio-conditioned generation at
        # all — must not fall back to the plain (non-a2v) matrix, or an unsupported
        # pipeline looks valid here and only fails downstream at the LTX API.
        if item.spec.a2v_supported_resolutions_durations is None:
            return None
        resolution_map = item.spec.a2v_supported_resolutions_durations
    else:
        resolution_map = item.spec.supported_resolutions_durations
    return resolution_map.get(resolution)


def get_supported_durations(
    resolution_spec: LTXVideoGenerationResolutionSpec,
    *,
    fps: LTXVideoGenFps,
) -> list[LTXVideoGenDuration]:
    return list(resolution_spec.fps_to_durations.get(fps, []))


def supported_duration_range(
    item: LTXVideoGenerationModelSpecItem,
    *,
    resolution: LTXVideoGenResolution,
    fps: LTXVideoGenFps,
    is_a2v: bool = False,
) -> tuple[LTXVideoGenDuration, LTXVideoGenDuration]:
    resolution_spec = _get_resolution_spec(item, resolution=resolution, is_a2v=is_a2v)
    if resolution_spec is None:
        raise KeyError(resolution)
    durations = get_supported_durations(resolution_spec, fps=fps)
    if not durations:
        raise KeyError(fps)
    return min(durations), max(durations)


VideoGenerationFeature = Literal[
    "multi-keyframe",
    "audio-to-video",
    "image-to-video",
    "text-to-video",
]


def video_generation_feature(
    *,
    audio_path: str | None,
    image_path: str | None,
    last_image_path: str | None = None,
    keyframes: object = None,
) -> VideoGenerationFeature:
    """User-facing mode. Keyframes win, then audio, then a still.

    Same order as ``videoGenerationModeFromInputs`` in the frontend. Validation
    error text and analytics feature names both use this so a keyframe job is
    not recorded as image-to-video.
    """
    if keyframes:
        return "multi-keyframe"
    if audio_path:
        return "audio-to-video"
    if image_path or last_image_path:
        return "image-to-video"
    return "text-to-video"


def validate_generate_video_request(
    req: GenerateVideoRequest,
    *,
    use_api_specs: bool,
    local_model_id: LTXLocalModelId | None = None,
    duration_head_ready: bool = False,
) -> str | None:
    items = (
        get_api_video_generation_model_specs()
        if use_api_specs
        else get_local_video_generation_model_specs(
            local_model_id, duration_head_ready=duration_head_ready
        )
    )
    item = next((candidate for candidate in items if candidate.pipeline == req.model), None)
    generation_backend = "api" if use_api_specs else "local"
    image_path = normalize_optional_path(req.imagePath)
    last_image_path = normalize_optional_path(req.lastImagePath)
    audio_path = normalize_optional_path(req.audioPath)
    generation_mode = video_generation_feature(
        audio_path=audio_path,
        image_path=image_path,
        last_image_path=last_image_path,
        keyframes=req.keyframes,
    )

    keyframe_error = validate_keyframe_inputs(
        req,
        use_api_specs=use_api_specs,
        image_path=image_path,
        last_image_path=last_image_path,
        audio_path=audio_path,
    )
    if keyframe_error is not None:
        return keyframe_error

    if last_image_path:
        if not image_path:
            return "Last frame requires a first-frame image"
        if req.duration is None:
            return "Last frame cannot be combined with automatic duration"

    if item is None:
        return (
            f"Unsupported {generation_backend} video generation pipeline: {req.model}. "
            f"Supported pipelines: {', '.join(candidate.pipeline for candidate in items)}"
        )

    resolution_spec = _get_resolution_spec(item, resolution=req.resolution, is_a2v=audio_path is not None)
    if resolution_spec is None:
        return (
            f"Unsupported {generation_backend} {generation_mode} resolution '{req.resolution}' "
            f"for pipeline '{req.model}'"
        )

    if req.aspectRatio not in resolution_spec.aspect_ratios:
        return (
            f"Unsupported {generation_backend} {generation_mode} aspect ratio '{req.aspectRatio}' "
            f"for pipeline '{req.model}' at resolution '{req.resolution}'"
        )

    if req.fps not in resolution_spec.fps_to_durations:
        return (
            f"Unsupported {generation_backend} {generation_mode} fps '{req.fps}' "
            f"for pipeline '{req.model}' at resolution '{req.resolution}'"
        )

    if req.duration is None:
        if audio_path is not None:
            return "Automatic duration cannot be combined with audio-to-video"
        if item.spec.capabilities is None or not item.spec.capabilities.auto_duration:
            return (
                f"Automatic duration is not supported for {generation_backend} "
                f"pipeline '{req.model}'"
            )
        return None

    supported_durations = get_supported_durations(resolution_spec, fps=req.fps)
    if req.duration not in supported_durations:
        return (
            f"Unsupported {generation_backend} {generation_mode} duration '{req.duration}' "
            f"for pipeline '{req.model}' at resolution '{req.resolution}' and fps '{req.fps}'"
        )

    if req.keyframes:
        caps = item.spec.capabilities
        if caps is None or not caps.multi_keyframe:
            return (
                f"Multi-keyframe is not supported for {generation_backend} "
                f"pipeline '{req.model}'"
            )
        if len(req.keyframes) > caps.multi_keyframe_max_count:
            return f"You can place up to {caps.multi_keyframe_max_count} keyframes"

    return None
