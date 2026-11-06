"""Token/VRAM estimator for local video jobs.

Tool-agnostic: Fast t2v/i2v/a2v sizes advertised stage-2 H×W (not half-res
stage-1). IC-LoRA wraps this with a ×2 reference sequence and a measured
spatial No.

Linear ``estimated_giB`` is intentionally a bit high so we 422 / stream early,
not late. Hang on a borderline mis-judge is acceptable.

Windows Fast (LTX 2.3 and 2.5; both ids are ``is_windows_fast``) gates on
max() of two continuous upper bounds: the allocated OOM law (capped runs can
only fit allocated — ``cuda_max_reserved`` expands to fill a virtual cap
while ``cuda_max_allocated`` is invariant) and the reserved paging law
(fitted to native reserved — on physical cards the reverse holds and
allocated-fitting cells page). Full vs stream selection still uses the
generic full curve with 2 GiB headroom.
Attention backend is deliberately not an availability input (measured no
difference). Do not gate on it without new data.

Mac (Darwin) LTX 2.5 Fast gates on the same continuous shape with Mac-native
coefficients. Compared against the sampled MPS pool, never full resident, no
RAM gate (rationales at decide_mac_job). Darwin 2.3 rides the same stream on
Windows-measured family equivalence (see is_mac_fast); unknown model ids fail
closed in is_mac_fast's assert-never until priced. Darwin additionally fences
single-forward sequences past the MPS kernel's element limit, a hard refusal
independent of the pool (see _MPS_TRANSFORMER_TOKEN_FENCE).
"""
# Darwin 2.3 routing evidence (for the module contract above): the Windows
# pair method plus the archived parity re-proofs.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Protocol, assert_never

from api_types import BudgetReadout, GenerationRisk, LTXLocalModelId
from frame_math import compute_num_frames
from runtime_config.fast_chunking import (
    PEAK_PLATEAU_WINDOWS,
    is_chunked_canvas,
    job_window_count,
    window_candidates,
    window_count,
)
from runtime_config.measured_cells import (
    BudgetFeature,
    BudgetRoute,
    is_unmeasured_high_res,
    is_unmeasured_windowed_below_1440p,
    window_fallback_measured,
)
from runtime_config.reject_reasons import (
    BudgetMode,
    JobRiskTier,
    LoadRiskTier,
    ShortfallKind,
    VideoJobRejectReason,
)
from runtime_config.runtime_policy import (
    CUDA_CPU_OFFLOAD_RAM_FLOOR_GB,
    LocalGenerationMode,
    windows_checkpoint_streams_from_disk,
)
from services.storage_info.storage_info import DriveType

if TYPE_CHECKING:
    # Runtime import would cycle (local_runtime imports this module).
    from runtime_config.runtime_config import RuntimeConfig

LoadMode = Literal["full_models_loading", "streaming_models_loading"]

VIDEO_JOB_TOO_LARGE = "VIDEO_JOB_TOO_LARGE"
VIDEO_JOB_TOO_LARGE_MESSAGE = (
    "This job is larger than this machine can run locally right now. "
    "Lower the resolution, duration, frame rate, or aspect width."
)
LOCAL_GENERATION_UNSUPPORTED = "LOCAL_GENERATION_UNSUPPORTED"
LOCAL_GENERATION_UNSUPPORTED_MESSAGE = "This machine cannot run local generation."
LOCAL_GENERATION_RAM_SHORTFALL = "LOCAL_GENERATION_RAM_SHORTFALL"
LOCAL_GENERATION_RAM_SHORTFALL_MESSAGE = (
    "This machine does not have enough system RAM for local generation."
)
# Fixed copy for the MPS sequence fence: nothing was compared, so no figures.
# Rides VIDEO_JOB_TOO_LARGE like the other fence verdicts.
MPS_SEQUENCE_OVER_MESSAGE = (
    "This job is too long for this Mac's GPU to run in one pass. "
    "Lower the resolution, duration, frame rate, or aspect width."
)
# Fixed copy for a 1440p or 4K cell no run has confirmed. Nothing was compared, so no figures.
UNMEASURED_CELL_MESSAGE = (
    "This resolution has not been measured on this machine, so it is not offered "
    "yet. Lower the resolution."
)
# Total over VideoJobRejectReason: a new reason must add a row here. Codes come
# from this table; messages come from reject_message_and_code (enriched with the
# measured numbers) with these rows as the incoherent/unknown fallback — never
# print a number the budget did not actually compare. All budget raise sites map
# rejects through that builder — no per-handler chains.
VIDEO_JOB_REJECT_ERRORS: dict[VideoJobRejectReason, tuple[str, str]] = {
    "unsupported": (LOCAL_GENERATION_UNSUPPORTED_MESSAGE, LOCAL_GENERATION_UNSUPPORTED),
    "ram_shortfall": (LOCAL_GENERATION_RAM_SHORTFALL_MESSAGE, LOCAL_GENERATION_RAM_SHORTFALL),
    "stream_over": (VIDEO_JOB_TOO_LARGE_MESSAGE, VIDEO_JOB_TOO_LARGE),
    "mps_sequence_over": (MPS_SEQUENCE_OVER_MESSAGE, VIDEO_JOB_TOO_LARGE),
    "unmeasured_cell": (UNMEASURED_CELL_MESSAGE, VIDEO_JOB_TOO_LARGE),
}

# Default shrink-job advice for stream_over total-short and Darwin pool
# messages. Home edit callers pass the resolution-only variant instead —
# queued Home pads context back up to the cap, so a shorter duration does
# not shrink that job, while GenSpace encodes the whole source, so it does.
_FULL_ACTION = "Lower the resolution, duration, frame rate, or aspect width."
# Default free-short advice: closing GPU apps frees VRAM and a retry can
# succeed. Queued Home callers override it — a Home re-pad makes retry
# futile, so the advice is the same resolution-only shrink as total-short.
_FREE_SHORT_ACTION = "Close other GPU apps and retry."


def reject_message_and_code(
    decision: VideoJobReject,
    job_filter: LocalJobFilter,
    *,
    action: str = _FULL_ACTION,
    free_short_action: str = _FREE_SHORT_ACTION,
) -> tuple[str, str]:
    """Enriched (message, code) for a budget reject, naming the binding constraint.

    Reads the reason and compared figures off ``decision``: its ``stream_gib``
    is the figure the gate actually compared; its ``required_ram_bytes`` is
    the floor, set only on ram_shortfall — never re-derived from a profile
    property. Unsupported short-circuits before any comparison; a reject
    whose budget figures are unknown falls back to the table message rather
    than printing a lie.

    ``stream_over`` splits free-short from total-short because min(total, free)
    gating makes those different user actions: free-short can retry after
    closing GPU apps, total-short must shrink the job. On Darwin the gate is
    the MPS pool (recommended_max, not free memory), so closing apps is not
    the advice there either. ``action`` is the shrink-job advice, chosen
    caller-side; ``free_short_action`` is the free-short advice, also chosen
    caller-side (queued Home overrides it: context re-pads, so retry is
    futile and the job must shrink instead).
    """
    reason = decision.reason
    required_gib = decision.stream_gib
    required_ram_bytes = decision.required_ram_bytes
    legacy, code = VIDEO_JOB_REJECT_ERRORS[reason]
    if reason == "unsupported":
        return (legacy, code)
    if reason == "mps_sequence_over" or reason == "unmeasured_cell":
        # Fence verdict: the pool figures were never compared, so fixed copy.
        return (legacy, code)
    if reason == "ram_shortfall":
        detected = job_filter.budget.total_ram_bytes
        if required_ram_bytes is None or detected is None:
            return (legacy, code)
        return (
            "This job needs "
            f"{required_ram_bytes // _GIB} GB of system RAM but this machine has "
            f"{detected // _GIB} GB. Close heavy programs; on desktops, more RAM "
            "is the real fix.",
            code,
        )
    budget = job_filter.budget
    if job_filter.profile.system == "Darwin":
        pool = budget.mps_pool_gib
        if pool is None:
            return (legacy, code)
        return (
            f"This job needs {required_gib:.1f} GB of GPU memory but this Mac allows "
            f"{pool:.1f} GB. {action}",
            code,
        )
    total = budget.total_vram_gib
    available = budget.available_vram_gib
    if total is None or available is None:
        return (legacy, code)
    if required_gib > total:
        return (
            f"This job needs {required_gib:.1f} GB of GPU memory but this card has "
            f"{total:.1f} GB. {action}",
            code,
        )
    if required_gib > available:
        # available < total here (total-short ruled out above), so available is
        # the free-VRAM figure, not the total fallback.
        return (
            f"This job needs {required_gib:.1f} GB of GPU memory but only "
            f"{available:.1f} GB is free right now (of {total:.1f} GB total). "
            f"{free_short_action}",
            code,
        )
    return (legacy, code)


def budget_readout_for_reject(
    decision: VideoJobReject, job_filter: LocalJobFilter
) -> BudgetReadout | None:
    """Need and have for the same comparison ``reject_message_and_code`` prints.

    None for a hard wall or when the gate compared no figure.
    """
    reason = decision.reason
    if reason == "unmeasured_cell":
        return untested_readout()
    if reason == "unsupported" or reason == "mps_sequence_over":
        return None
    if reason == "ram_shortfall":
        required = decision.required_ram_bytes
        detected = job_filter.budget.total_ram_bytes
        if required is None or detected is None:
            return None
        return BudgetReadout(
            kind="ram",
            need_gb=float(required // _GIB),
            have_gb=float(detected // _GIB),
        )
    required_gib = decision.stream_gib
    if job_filter.profile.system == "Darwin":
        pool = job_filter.budget.mps_pool_gib
        if pool is None:
            return None
        return BudgetReadout(kind="mac", need_gb=required_gib, have_gb=pool)
    total = job_filter.budget.total_vram_gib
    available = job_filter.budget.available_vram_gib
    if total is None or available is None:
        return None
    if required_gib > total:
        return BudgetReadout(kind="gpu", need_gb=required_gib, have_gb=total)
    if required_gib > available:
        return BudgetReadout(kind="gpu_busy", need_gb=required_gib, have_gb=available)
    return None


def tight_readout(ask_gib: float, available_gib: float) -> BudgetReadout:
    return BudgetReadout(kind="tight", need_gb=ask_gib, have_gb=available_gib)


def untested_readout() -> BudgetReadout:
    return BudgetReadout(kind="untested")


TIGHT_FIT_HEADROOM_GIB = 0.5
TIGHT_FIT_HEADROOM_RATIO = 0.05


def compared_available_gib(job_filter: LocalJobFilter) -> float | None:
    """The figure the gate compared the ask against.

    Single home for the figure-selection rule: the MPS pool on Darwin,
    min(total, free) — or total under total_only — on CUDA. None means the
    gate compared nothing (callers stay silent).
    """
    if job_filter.profile.system == "Darwin":
        return job_filter.budget.mps_pool_gib
    if job_filter.budget.total_vram_bytes is None:
        return None
    return job_filter.budget.available_vram_gib


def load_headroom_gib(
    decision: VideoJobLoad, job_filter: LocalJobFilter
) -> float | None:
    """Available minus ask on the compared figure. None when uncompared."""
    available = compared_available_gib(job_filter)
    if available is None:
        return None
    return available - decision.stream_gib


def risk_allowed(allow_risk: bool | None, job_filter: LocalJobFilter) -> bool:
    """Whether an over-budget verdict downgrades to allow-with-warning.

    Tri-state: None follows the filter's live mode (every request-scoped gate
    passes None); executors pass an explicit bool from the stored annotation so
    a mode flip never re-gates a queued job. Named ``allow_risk`` (not
    ``risk_allowance``): ``risk_allowance`` is the plain bool carried on
    executor protocols from the stored spec annotation, while ``allow_risk``
    is the tri-state gate input that defaults to the live mode when no stored
    value exists.
    """
    return allow_risk if allow_risk is not None else job_filter.budget_mode == "reckless"


def risk_if_allowed(
    risk: GenerationRisk | None,
    job_filter: LocalJobFilter,
    allow_risk: bool | None,
    *,
    unmeasured_bet: bool = False,
) -> GenerationRisk | None:
    """The bet a refusal may run as, or None when the refusal stands.

    An ``over`` bet runs only when risk is allowed (live mode, or the stored
    allowance at execute). An ``unmeasured`` bet also runs when the stored
    annotation was unmeasured: that note covers the unmeasured refusal it was
    taken for, never a later VRAM or RAM miss, which still reports as ``over``.
    """
    if risk is None:
        return None
    if risk.tier == "unmeasured" and unmeasured_bet:
        return risk
    return risk if risk_allowed(allow_risk, job_filter) else None


def load_risk_tier(
    decision: VideoJobLoad,
    job_filter: LocalJobFilter,
    *,
    feature: BudgetFeature,
) -> LoadRiskTier | None:
    """Warning tier for an admitted job, if any.

    Legacy-route loads are unmeasured (admitted with no trustworthy model);
    measured loads inside max(0.5 GiB, 5% of ask) are tight fits. Unknown
    figures stay silent.
    """
    if budget_route(feature, job_filter.profile) == "legacy":
        return "unmeasured"
    headroom = load_headroom_gib(decision, job_filter)
    if headroom is None:
        return None
    threshold = max(
        TIGHT_FIT_HEADROOM_GIB, TIGHT_FIT_HEADROOM_RATIO * decision.stream_gib
    )
    return "tight" if headroom < threshold else None


def reject_risk_tier(reason: VideoJobRejectReason) -> JobRiskTier | None:
    """Measurement rejects are over-budget bets, an unmeasured cell is an
    unmeasured bet; hard walls return None.

    Allow-list: a new ``VideoJobRejectReason`` fails typecheck here until it
    picks a tier. ``unsupported`` (no pipeline) and ``mps_sequence_over`` (the
    Metal kernel SIGABRTs past the fence — uncatchable, not a bet) never
    downgrade.
    """
    match reason:
        case "stream_over" | "ram_shortfall":
            return "over"
        case "unmeasured_cell":
            return "unmeasured"
        case "unsupported" | "mps_sequence_over":
            return None
        case _ as unreachable:
            assert_never(unreachable)


def reject_over_gib(
    decision: VideoJobReject, job_filter: LocalJobFilter
) -> float | None:
    """VRAM shortfall on a stream_over reject. None for any other reason —
    RAM verdicts passed VRAM, so subtracting would print a negative lie."""
    if decision.reason != "stream_over":
        return None
    available = compared_available_gib(job_filter)
    if available is None:
        return None
    return decision.stream_gib - available


def reject_over_ram_gb(
    decision: VideoJobReject, job_filter: LocalJobFilter
) -> int | None:
    """RAM shortfall in whole GB, message-consistent (required//GiB minus
    detected//GiB, the same ints the 422 prints)."""
    if decision.reason != "ram_shortfall":
        return None
    required = decision.required_ram_bytes
    detected = job_filter.budget.total_ram_bytes
    if required is None or detected is None:
        return None
    return required // _GIB - detected // _GIB


def reject_shortfall_kind(
    decision: VideoJobReject, job_filter: LocalJobFilter
) -> ShortfallKind | None:
    """Which figure bound a CUDA stream_over reject.

    "free" only when free strictly binds (free < total on exact floats —
    min() returns its input, so equality with total is exact). Total-gated
    lists, missing samples, and Darwin (pool gate, never free advice) all
    report "total"/None respectively: Darwin returns None and the frontend
    keys Darwin copy off platform, not kind.
    """
    if decision.reason != "stream_over":
        return None
    if job_filter.profile.system == "Darwin":
        return None
    total = job_filter.budget.total_vram_gib
    available = job_filter.budget.available_vram_gib
    if total is None or available is None:
        return None
    return "total" if available == total else "free"


def tight_fit_message(ask_gib: float, available_gib: float, *, darwin: bool) -> str:
    if darwin:
        return (
            f"Tight fit: this needs ~{ask_gib:.1f} GB of your Mac's "
            f"{available_gib:.1f} GB memory allowance. It should run."
        )
    return (
        f"Tight fit: this needs ~{ask_gib:.1f} GB and about "
        f"{available_gib:.1f} GB is available. It should run, but close "
        "other GPU apps first."
    )


def over_budget_message(
    ask_gib: float,
    available_gib: float,
    *,
    darwin: bool,
    shortfall_kind: ShortfallKind | None,
) -> str:
    if darwin:
        return (
            f"Over budget: this needs {ask_gib:.1f} GB but this Mac allows "
            f"{available_gib:.1f} GB. It may run out of memory or stall — "
            "lower the resolution or duration to fit."
        )
    if shortfall_kind == "free":
        return (
            f"Over budget right now: this needs {ask_gib:.1f} GB but only "
            f"{available_gib:.1f} GB is free. Close other GPU apps and retry, "
            "or lower the resolution or duration."
        )
    return (
        f"Over budget: this needs {ask_gib:.1f} GB but this machine's graphics "
        f"card has {available_gib:.1f} GB. It may run out of memory or stall — "
        "lower the resolution or duration to fit."
    )


def over_ram_message(required_gb: int, detected_gb: int) -> str:
    return (
        f"Over budget: this needs {required_gb} GB of system RAM but this machine "
        f"has {detected_gb} GB. Close heavy programs; on desktops, more RAM "
        "is the real fix."
    )


def unmeasured_message(
    hint: Literal["mac_two_stage"] | None = None,
) -> str:
    if hint == "mac_two_stage":
        return (
            "Unmeasured: two-stage IC-LoRA isn't measured on Mac yet. "
            "It may fail or stall."
        )
    return (
        "Unmeasured: we haven't tested this setup on your hardware, so we "
        "can't predict whether it fits. It may fail or stall."
    )


def unmeasured_load_risk(
    decision: VideoJobLoad,
    job_filter: LocalJobFilter,
    *,
    feature: BudgetFeature,
    allow_risk: bool | None,
) -> GenerationRisk | None:
    """T3 annotation for a legacy load. Reckless only.

    Tight fits stay unmarked. Standard stays unmarked. A stored unmeasured
    note does not waive a later VRAM or RAM miss at execute.
    """
    if not risk_allowed(allow_risk, job_filter):
        return None
    if load_risk_tier(decision, job_filter, feature=feature) != "unmeasured":
        return None
    return GenerationRisk(
        tier="unmeasured",
        message=unmeasured_message(),
        readout=untested_readout(),
    )


def risk_for_reject(
    decision: VideoJobReject, job_filter: LocalJobFilter
) -> GenerationRisk | None:
    """Over-budget annotation for a measurement reject, if downgradable.

    Returns None for unsupported (hard wall) and for uncompared figures.
    Legacy-route rejects tier as over: the curve is rough, but a reject is
    conservative by construction — the unmeasured bet is the legacy *load*.
    """
    tier = reject_risk_tier(decision.reason)
    if tier is None:
        return None
    if tier == "unmeasured":
        return GenerationRisk(
            tier="unmeasured",
            message=unmeasured_message(),
            readout=untested_readout(),
        )
    darwin = job_filter.profile.system == "Darwin"
    if decision.reason == "ram_shortfall":
        required = decision.required_ram_bytes
        detected = job_filter.budget.total_ram_bytes
        if required is None or detected is None:
            return None
        required_gb = required // _GIB
        detected_gb = detected // _GIB
        return GenerationRisk(
            tier="over",
            shortfall_kind=None,
            over_gib=None,
            over_ram_gb=required_gb - detected_gb,
            message=over_ram_message(required_gb, detected_gb),
            readout=budget_readout_for_reject(decision, job_filter),
        )
    available = compared_available_gib(job_filter)
    if available is None:
        return None
    kind = reject_shortfall_kind(decision, job_filter)
    over = decision.stream_gib - available
    return GenerationRisk(
        tier="over",
        shortfall_kind=kind,
        over_gib=over,
        over_ram_gb=None,
        message=over_budget_message(
            decision.stream_gib, available, darwin=darwin, shortfall_kind=kind
        ),
        readout=budget_readout_for_reject(decision, job_filter),
    )


def vram_snapshot_suffix(
    *,
    allocated_bytes: int | None,
    reserved_bytes: int | None,
    free_bytes: int | None,
) -> str:
    """Gate-time VRAM triple for budget log lines.

    ``allocated``/``reserved`` are live torch probes; ``free`` is the figure
    the gate actually compared (the filter's sample, never re-sampled for
    logging). Reserved minus allocated is dead torch cache: the next triage
    reads reclaimable occupancy off that gap instead of guessing.
    """
    def _gib_or_unknown(value_bytes: int | None) -> str:
        return f"{value_bytes / _GIB:.1f}GiB" if value_bytes is not None else "unknown"

    return (
        f" (vram alloc={_gib_or_unknown(allocated_bytes)}"
        f" reserved={_gib_or_unknown(reserved_bytes)}"
        f" free={_gib_or_unknown(free_bytes)})"
    )

# Legacy full curve: predates the measured rows below. Re-derive rather than trust.
_FULL_INTERCEPT_GIB = 18.0
_FULL_GIB_PER_TOKEN = 0.000244
# Legacy stream curve: small cells fit small cards, large cells 422 instead of
# paging. Same basis as the full curve above. Re-derive rather than trust.
_STREAM_INTERCEPT_GIB = 3.1
_STREAM_GIB_PER_TOKEN = 0.000315
# A GPU that reports 32 would otherwise pick full and page. Keep this off the
# full-mode comparison only.
_FULL_VRAM_HEADROOM_GIB = 2.0
# Edit full-loads page where Fast fits: a 720p retake reserved the whole card
# and paged while the same card streamed 1080p. 5 GiB so that clip streams on a
# card that reports 31 or 32. Do not drop this to 4: a 32-report would
# full-load the clip again.
_EDIT_FULL_HEADROOM_GIB = 5.0
# Do not round the stream intercept/slope without re-logging 16 GB and 24 GB cards.

_GIB = 1024**3
# VRAM floors sit below nominal because cards report low; RAM floors are nominal,
# so the comparison carries the headroom instead. 1 GiB covers the observed
# ~0.5 GiB Windows under-report with margin. Messages still print nominal —
# only the verdict compares with headroom.
_RAM_REPORTING_HEADROOM_BYTES = 1 * _GIB
# Streamed jobs pin tens of GiB of transformer in host RAM. Shared with the
# Linux CPU-offload floor: same working set, same number — only the consequence
# differs (Linux streams from disk below it; Windows declines the cell on the
# CPU path and maps from disk on NVMe).
WINDOWS_FAST_STREAMING_RAM_BYTES = CUDA_CPU_OFFLOAD_RAM_FLOOR_GB * _GIB
# Disk-mapped streaming commits little host RAM, but the floor stays well above
# that: with less room the page cache thins and mapped reads balloon per run.
# 32 GB is the lowest figure the evidence supports. Below it stays refused
# until measured.
# Basis note: the low-RAM figures trace only to brief prose
WINDOWS_FAST_DISK_STREAMING_RAM_GB = 32
# Edit RAM floors: the CPU warm pin exceeds what a 48 GB machine can hold plus
# the OS, so the CPU floor is 64; DISK holds 32 by the Fast precedent (less
# degraded there than Fast at its own floor).
WINDOWS_EDIT_CPU_STREAMING_RAM_GB = 64
WINDOWS_EDIT_DISK_STREAMING_RAM_GB = 32
# IC-LoRA RAM floors: reference latents lift the CPU warm pin over Fast's, so
# the CPU floor is 64; DISK degrades steeply at 40 but holds at 48.
WINDOWS_ICLORA_CPU_STREAMING_RAM_GB = 64
WINDOWS_ICLORA_DISK_STREAMING_RAM_GB = 48


@dataclass(frozen=True, slots=True)
class _StreamModel:
    """One floored-linear stream requirement: max(floor, intercept + slope*tokens).

    The floor covers the resident/encode region where the line does not apply;
    the slope folds working set plus the per-token transient.
    """

    floor_gib: float
    intercept_gib: float
    gib_per_token: float

    def required_gib(self, tokens: float) -> float:
        return max(
            self.floor_gib, self.intercept_gib + self.gib_per_token * tokens
        )


# Windows Fast stream requirement: the transient is a per-token allocation, not
# a per-resolution-class constant, so the slope folds it into the allocated fit.
# An upper bound, not a floor. This row predicts OOM crashes; the gate takes
# max() with the reserved row below, which predicts paging.
_WINDOWS_FAST_STREAM = _StreamModel(
    floor_gib=8.00,
    intercept_gib=1.552,
    gib_per_token=2.6481e-4,
)
# Windows Fast RESERVED row: native streamed reserved breaches small cards the
# allocated row offers. Same paging failure as edit. Fitted on native runs
# only — caps force reserved to equal the cap and destroy the signal. Floor is
# the encode-region ceiling; 10.00 would refuse idle 10 GB cards.
# Near-twin of the edit reserved law: same allocator physics, fitted
# independently per envelope — do not merge the rows.
_WINDOWS_FAST_RESERVED_STREAM = _StreamModel(
    floor_gib=9.90,
    intercept_gib=5.00,
    gib_per_token=2.80e-4,
)
# Windows Fast FULL-mode reserved row: the legacy full ask under-calls native
# full-mode reserved on every measured cell — and allocated too (allocated sits
# flat while reserved grows) — so the full pick admitted paging jobs. Fitted on
# native full-mode runs only. The law dominates the legacy ask at every token
# count, so it always binds.
_WINDOWS_FAST_FULL_RESERVED = _StreamModel(
    floor_gib=23.70,
    intercept_gib=20.90,
    gib_per_token=3.10e-4,
)
# Edit (retake/extend) gates on native reserved: an allocated-fitting retake
# paged past physical VRAM. Allocated predicts crashes; reserved predicts paging.
# Only the floor keeps its old basis: small-canvas edit fails where Fast passes
# (fragmentation from resident source latents, not bigger tensors).
# Unlike Fast's max(allocated, reserved) form, this row replaces: reserved
# strictly dominates the allocated line everywhere above the floor, so max()
# would be the same function — one row, less machinery. Revisit if a refit
# un-dominates.
_EDIT_STREAM = _StreamModel(
    floor_gib=8.75,
    intercept_gib=5.00,
    gib_per_token=2.77e-4,
)
# Mac (Darwin) Fast stream requirement, same shape as Windows, Mac-native fit.
# The per-token transient transfers from CUDA; the Mac surcharge is working set
# only. An upper bound, not a floor.
_MAC_FAST_STREAM = _StreamModel(
    floor_gib=9.75,
    intercept_gib=3.56,
    gib_per_token=3.349e-4,
)
# Mac (Darwin) edit stream requirement, Mac-native fit: retake == extend, one
# envelope serves both. Fragmentation transfers from Windows; only
# floor/intercept shift on MPS.
_MAC_EDIT_STREAM = _StreamModel(
    floor_gib=9.27,
    intercept_gib=7.10,
    gib_per_token=2.669e-4,
)
# Public scalar for matrix prose. Single source: the row above.
MAC_FAST_FLOOR_GIB = _MAC_FAST_STREAM.floor_gib
# Nominal MPS pool for compat-matrix Darwin bands: min observed ratio. The ratio
# is SKU-dependent, so the min is the conservative display pick.
# Display-only: the runtime gate probes the real pool per request.
NOMINAL_MPS_POOL_RATIO = 0.78


def video_tokens(height: int, width: int, frames: int) -> float:
    """LTX-2 latent tokens: 8-frame temporal, 32 px spatial."""
    temporal = (max(1, frames) - 1) / 8.0 + 1.0
    return temporal * (height / 32.0) * (width / 32.0)


def estimated_giB(mode: LoadMode, seq: float) -> float:
    if mode == "full_models_loading":
        return _FULL_INTERCEPT_GIB + _FULL_GIB_PER_TOKEN * seq
    return _STREAM_INTERCEPT_GIB + _STREAM_GIB_PER_TOKEN * seq


def fits(seq: float, mode: LoadMode, memory_gb: float) -> bool:
    return estimated_giB(mode, seq) <= memory_gb


def _full_memory_gb(memory_gb: float, headroom_gib: float = _FULL_VRAM_HEADROOM_GIB) -> float:
    return max(0.0, memory_gb - headroom_gib)


def memory_gb_for_job(
    *,
    vram_gb: float | None,
    available_ram_gb: float | None,
    darwin: bool,
) -> float | None:
    """Discrete VRAM on CUDA; free RAM on Darwin (already not total)."""
    if darwin:
        return available_ram_gb
    return vram_gb


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """What machine + model + mode a budget decision is evaluated for.

    Profile facts for budget dispatch. The fused route lives in
    ``budget_route`` (keyed by feature, system, model);
    ``is_windows_fast`` / ``is_mac_fast`` are the Fast model predicates it
    consults (and handlers consult for preflight gating). Nothing downstream
    compares ``system`` / ``model_id`` strings directly.
    """

    system: Literal["Windows", "Linux", "Darwin"]
    model_id: LTXLocalModelId | None
    process_mode: LocalGenerationMode
    # Drive class of the checkpoint volume, probed once at startup. None means
    # unknown (unimplemented platform, probe failure) — never "slow".
    drive_type: DriveType | None = None
    # Native fp8 (compute capability >= 8.9). The measured laws were fit on
    # fp8; without it the budget takes the legacy curve.
    fp8_capable: bool = True

    @property
    def darwin(self) -> bool:
        """Whether this profile runs on Darwin (derived from ``system``)."""
        return self.system == "Darwin"

    @property
    def is_windows_fast(self) -> bool:
        """2.5 measured; 2.3 shares those bands.

        Same-clip pairs put 2.3 and 2.5 within a few hundredths of a GiB, the
        same family-equivalence Mac Fast already routes on. Both 2.3 checkpoint
        ids are included. Anything else on Windows stays on the generic curve.
        """
        return self.system == "Windows" and self.model_id in (
            "ltx-2.5-22b-distilled",
            "ltx-2.3-22b-distilled",
            "ltx-2.3-22b-distilled-1.1",
        )

    # Darwin 2.3 routing evidence (for is_mac_fast below): the Windows pair
    # method plus the archived parity re-proofs.
    @property
    def is_mac_fast(self) -> bool:
        """Darwin Fast routes 2.5 (measured) and 2.3 (family-equivalence extension).

        The 2.3-vs-2.5 equivalence is Windows-measured only; routing Darwin 2.3
        through _MAC_FAST_STREAM is extrapolation. The alternative is the legacy
        Darwin branch, which never refuses at all. A future model id fails
        closed via assert_never until it is priced — it must never fall through
        to the legacy never-reject branch silently.
        """
        if self.system != "Darwin":
            return False
        if self.model_id is None:
            return False
        match self.model_id:
            case "ltx-2.3-mlx-q4":
                return False
            case (
                "ltx-2.5-22b-distilled"
                | "ltx-2.3-22b-distilled"
                | "ltx-2.3-22b-distilled-1.1"
            ):
                return True
            case _ as unreachable:
                assert_never(unreachable)

    @property
    def streams_from_disk(self) -> bool:
        """Windows + NVMe maps weights from disk; everything else pins from CPU.

        Reads the shared rule, so the RAM floors below price exactly the
        DISK path the offload routing takes.
        """
        return windows_checkpoint_streams_from_disk(self.system, self.drive_type)

    @property
    def windows_fast_required_ram_bytes(self) -> int:
        """RAM floor gating this profile's streamed Windows Fast jobs.

        Single source of truth shared by the budget gate and the reported
        ``local_unavailable_required_ram_gb``: both read the floor from the
        profile that produced the verdict. Mirrors the offload routing — the
        gate only runs for streaming-selected jobs, so Windows + NVMe here is
        exactly the DISK path there, and everything else pins from CPU.
        """
        if self.streams_from_disk:
            return WINDOWS_FAST_DISK_STREAMING_RAM_GB * _GIB
        return WINDOWS_FAST_STREAMING_RAM_BYTES

    @property
    def windows_edit_required_ram_bytes(self) -> int:
        """RAM floor gating this profile's streamed Windows edit jobs.

        See the WINDOWS_EDIT_* constants: Windows + NVMe maps from disk,
        everything else pins from CPU. Read only by decide_edit_job's Windows
        gate.
        """
        if self.streams_from_disk:
            return WINDOWS_EDIT_DISK_STREAMING_RAM_GB * _GIB
        return WINDOWS_EDIT_CPU_STREAMING_RAM_GB * _GIB

    @property
    def windows_iclora_required_ram_bytes(self) -> int:
        """RAM floor gating this profile's streamed Windows IC-LoRA jobs.

        See the WINDOWS_ICLORA_* constants: Windows + NVMe maps from disk,
        everything else pins from CPU. Read only by decide_ic_lora_job's
        Windows gate.
        """
        if self.streams_from_disk:
            return WINDOWS_ICLORA_DISK_STREAMING_RAM_GB * _GIB
        return WINDOWS_ICLORA_CPU_STREAMING_RAM_GB * _GIB


def budget_route(feature: BudgetFeature, profile: HardwareProfile) -> BudgetRoute:
    """Fused measured/legacy dispatch keyed by (feature, system, model).

    Single place that selects the stream curve. Route names carry the
    measured flag explicitly: ``mac_measured`` / ``windows_measured`` take
    fitted curves, ``legacy`` takes the generic path.     Fast on Darwin
    consults ``is_mac_fast`` (unknown model ids fail closed there);
    Fast on Windows is 2.5 and 2.3 via ``is_windows_fast``; edit and IC-LoRA
    are measured on Windows and Darwin for any model. Process floors
    (``CUDA_VRAM_FLOOR_GB``, ``CUDA_FULL_VRAM_GB``, ``DARWIN_*``) and the
    per-feature RAM floors stay where they are — this route only selects
    the curve, never the floor.
    """
    if feature == "fast":
        if profile.system == "Darwin":
            return "mac_measured" if profile.is_mac_fast else "legacy"
        if profile.system == "Windows":
            if not profile.fp8_capable:
                return "legacy"
            return "windows_measured" if profile.is_windows_fast else "legacy"
        if profile.system == "Linux":
            return "legacy"
        assert_never(profile.system)
    if feature == "edit":
        if profile.system == "Windows":
            return "legacy" if not profile.fp8_capable else "windows_measured"
        if profile.system == "Darwin":
            return "mac_measured"
        if profile.system == "Linux":
            return "legacy"
        assert_never(profile.system)
    if feature == "iclora":
        if profile.system == "Windows":
            return "legacy" if not profile.fp8_capable else "windows_measured"
        if profile.system == "Darwin":
            return "mac_measured"
        if profile.system == "Linux":
            return "legacy"
        assert_never(profile.system)
    assert_never(feature)


def _floored_gib(memory_gb: float | None) -> float | None:
    """Coarse integer-GB figure as GiB, floored to byte granularity."""
    if memory_gb is None:
        return None
    return int(memory_gb * _GIB) / _GIB


def _total_vram_gib(
    total_vram_bytes: int | None, memory_gb: float | None
) -> float | None:
    """Total VRAM figure: exact bytes win, the integer falls back.

    Both-None fails closed (None) — the call site refuses as unsupported.
    """
    if total_vram_bytes is not None:
        return total_vram_bytes / _GIB
    return _floored_gib(memory_gb)


def _available_vram_gib(
    total_gib: float | None, free_vram_bytes: int | None
) -> float | None:
    """Consumable VRAM figure: min(total, free).

    Free-None fails open to total (unmeasured free never shrinks the offer);
    total-None fails closed (None).
    """
    if total_gib is None:
        return None
    if free_vram_bytes is None:
        return total_gib
    return min(total_gib, free_vram_bytes / _GIB)


def _mps_pool_gib(mps_pool_bytes: int | None) -> float | None:
    """MPS allocator pool in GiB, exact bytes.

    Pool-None fails closed (None) with no integer fallback — unlike free
    VRAM, there is no total to fall back to.
    """
    if mps_pool_bytes is None:
        return None
    return mps_pool_bytes / _GIB


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """VRAM + RAM figures plus their precedence, evaluated once per use.

    Sampling: VRAM total is startup-frozen device bytes; free VRAM is sampled
    fresh per request via ``GpuInfo.get_free_cuda_vram_bytes``; RAM total is the
    startup-frozen machine total via ``psutil.virtual_memory().total``; the
    MPS pool is sampled fresh per request via
    ``GpuInfo.get_mps_pool_bytes``. ``memory_gb`` is the coarse floored
    figure (CUDA integer GB / Darwin free RAM) backing the integer fallback.
    Precedence lives here so no caller re-derives it: the constructors above
    own the typed fail-mode asymmetry (free-None opens to total, pool-None
    closes, both-None closes), and unknown-everything fails closed at the
    call site. ``total_only`` gates CUDA availability on total ("will it ever
    fit") instead of min(total, free): queued creates and list-time display
    admit total-fitting cells while free is momentarily depressed, and the
    execute-time re-check — still free-gated — refuses genuine drift. It is
    not free-None (unmeasured): the sample is still taken, and the flag names
    the policy. Mac pool gating is unchanged.
    """

    total_vram_bytes: int | None
    free_vram_bytes: int | None
    total_ram_bytes: int | None
    memory_gb: float | None
    mps_pool_bytes: int | None
    total_only: bool = False

    @property
    def total_vram_gib(self) -> float | None:
        """Exact total VRAM, else the integer fallback. None only if both missing."""
        return _total_vram_gib(self.total_vram_bytes, self.memory_gb)

    @property
    def available_vram_gib(self) -> float | None:
        """What a new job may actually consume: min(total, free), or total when total_only."""
        if self.total_only:
            return self.total_vram_gib
        return _available_vram_gib(self.total_vram_gib, self.free_vram_bytes)

    def streaming_ram_state(
        self, *, required_bytes: int
    ) -> Literal["ok", "shortfall", "unknown"]:
        """Tri-state RAM verdict: only known-insufficient claims shortfall."""
        if self.total_ram_bytes is None:
            return "unknown"
        return "ok" if self.total_ram_bytes >= required_bytes - _RAM_REPORTING_HEADROOM_BYTES else "shortfall"

    @property
    def mps_pool_gib(self) -> float | None:
        """MPS allocator pool in GiB, exact bytes. None off-MPS / unprobed."""
        return _mps_pool_gib(self.mps_pool_bytes)


@dataclass(frozen=True, slots=True)
class LocalJobFilter:
    """Profile + budget for list-time Fast specs and request-time job budget.

    Built once per request (one free-VRAM sample) and passed whole; nothing
    re-derives memory precedence or profile dispatch downstream.
    """

    profile: HardwareProfile
    budget: MemoryBudget
    # Live budget mode at build time. Gates read this (never global state)
    # to decide raise-vs-annotate; decide_* stays pure and mode-blind.
    budget_mode: BudgetMode = "standard"

    def decide(self, width: int, height: int, frames: int) -> VideoJobDecision:
        return decide_video_job(width, height, frames, job_filter=self)


def build_local_job_filter(
    config: RuntimeConfig,
    *,
    model_id: LTXLocalModelId | None,
    free_vram_bytes: int | None,
    mps_pool_bytes: int | None,
    total_only: bool = False,
    budget_mode: BudgetMode = "standard",
) -> LocalJobFilter:
    """One construction idiom for request filters. Callers sample probes first.

    ``None`` means unmeasured. Callers pass ``GpuInfo.get_free_cuda_vram_bytes()``
    and ``GpuInfo.get_mps_pool_bytes()`` sampled once per request; generic-curve
    paths pass None rather than paying for samples they ignore.

    ``total_only`` gates CUDA availability on total VRAM ("will it ever fit")
    instead of min(total, free): queued creates and list-time display admit
    total-fitting cells while free is momentarily depressed, and the
    execute-time re-check — still free-gated — refuses genuine drift with
    JOB_TOO_LARGE. It is not free-None (unmeasured): the sample is still
    taken, and the flag names the policy. GenSpace direct, estimate, and all
    execute checks keep the default free gate — branch at the call site, never
    here. Mac pool gating is unchanged.
    """
    # Unknown systems take the generic path, same as Linux: no measured
    # profile dispatches on them.
    if config.system == "Windows":
        system: Literal["Windows", "Linux", "Darwin"] = "Windows"
    elif config.system == "Darwin":
        system = "Darwin"
    else:
        system = "Linux"
    darwin = system == "Darwin"
    return LocalJobFilter(
        profile=HardwareProfile(
            system=system,
            model_id=model_id,
            process_mode=config.local_generations_mode,
            drive_type=config.checkpoint_drive_type,
            fp8_capable=config.fp8_capable,
        ),
        budget=MemoryBudget(
            total_vram_bytes=config.vram_bytes,
            free_vram_bytes=free_vram_bytes,
            total_ram_bytes=config.total_ram_bytes,
            memory_gb=memory_gb_for_job(
                vram_gb=config.vram_gb,
                available_ram_gb=config.available_ram_gb,
                darwin=darwin,
            ),
            mps_pool_bytes=mps_pool_bytes,
            total_only=total_only,
        ),
        budget_mode=budget_mode,
    )


@dataclass(frozen=True, slots=True)
class VideoJobLoad:
    mode: LoadMode
    seq: float
    full_gib: float
    stream_gib: float
    # Window length in pixel frames when the job runs in windows, else None (one pass).
    # ``seq`` and the GiB figures then describe one window. The pipeline runs this value.
    chunk_pixel_frames: int | None = None


@dataclass(frozen=True, slots=True)
class VideoJobReject:
    seq: float
    full_gib: float
    stream_gib: float
    reason: VideoJobRejectReason = "stream_over"
    # RAM floor the gate compared, set only on ram_shortfall (its sole
    # producer is _streaming_ram_reject, which holds the figure). Message
    # builders read this — never re-derive the floor from the profile, or a
    # wrong property prints a number the gate did not compare.
    required_ram_bytes: int | None = None
    # Set by IC-LoRA and by a Fast canvas that runs in windows, to the window the
    # pricing tried. A reckless downgrade must run that window: ``None`` is one
    # unchunked pass, which costs more than the over-budget figure the user
    # confirmed. Edit leaves it None.
    chunk_pixel_frames: int | None = None


VideoJobDecision = VideoJobLoad | VideoJobReject


def decide_from_fits(
    seq: float,
    full_gib: float,
    stream_gib: float,
    *,
    stream_fits: bool,
    full_fits: bool,
    process_mode: LocalGenerationMode,
    legacy_darwin_never_reject: bool,
) -> VideoJobDecision:
    """Single decision engine shared by the generic and Windows Fast paths.

    Callers evaluate their own fits (token curve vs allocated model) and the
    engine applies process viability, the legacy Darwin never-422 policy, and
    the full-vs-stream pick exactly once. Unknown budgets never reach here:
    callers map unknown to stream (generic) or fail closed (Windows) before
    calling. Measured Darwin profiles pass legacy_darwin_never_reject=False:
    the legacy branch is the absence of a ceiling, which they replace.
    """
    if process_mode == "unsupported":
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="unsupported"
        )

    if legacy_darwin_never_reject:
        return VideoJobLoad(
            mode="streaming_models_loading",
            seq=seq,
            full_gib=full_gib,
            stream_gib=stream_gib,
        )

    if not stream_fits:
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="stream_over"
        )

    if process_mode != "full_models_loading" or not full_fits:
        return VideoJobLoad(
            mode="streaming_models_loading",
            seq=seq,
            full_gib=full_gib,
            stream_gib=stream_gib,
        )
    return VideoJobLoad(
        mode="full_models_loading", seq=seq, full_gib=full_gib, stream_gib=stream_gib
    )


# Full-resident policy evidence (for the never-full rationale in
# decide_load_mode below): interleaved DISK-vs-NONE runs.
def decide_load_mode(
    seq: float,
    *,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    darwin: bool = False,
    full_headroom_gib: float = _FULL_VRAM_HEADROOM_GIB,
) -> VideoJobDecision:
    """422 definite-over jobs; per-job stream when full will not fit.

    Does not change process ``local_generations_mode``. ``full_headroom_gib`` is
    subtracted only from the full-vs-stream pick, not from the stream-over 422.

    Two Darwin policies (do not fold them together later):

    - Never full: MPS full-resident is only slightly faster at multiples of the
      shared unified-memory pool, and cannot serve smaller Macs at all
      (see ``runtime_policy``; ``decide_mac_job`` pins full_fits=False).
      The green/gray corruption that first ruled full out is fixed by
      ``ltx_core.devices.allow_async_transfer``.
      Independent of the token curve.
    - Never stream-over 422: the 5090 reserved curve is not unified memory, and
      ``available_ram_gb`` is already free-at-launch. Comparing them 422'd M4
      48 GB jobs that streamed. Fast 2.5/2.3 now has a Darwin-native pool ceiling
      (see ``is_mac_fast``); the legacy curve still never compares.

    Unknown memory streams rather than guessing full.
    """
    full_gib = estimated_giB("full_models_loading", seq)
    stream_gib = estimated_giB("streaming_models_loading", seq)

    if memory_gb is None:
        stream_fits, full_fits = True, False
    else:
        stream_fits = fits(seq, "streaming_models_loading", memory_gb)
        full_fits = fits(seq, "full_models_loading", _full_memory_gb(memory_gb, full_headroom_gib))
    return decide_from_fits(
        seq,
        full_gib,
        stream_gib,
        stream_fits=stream_fits,
        full_fits=full_fits,
        process_mode=process_mode,
        legacy_darwin_never_reject=darwin,
    )


def gate_unmeasured_cell(
    decision: VideoJobDecision,
    *,
    feature: BudgetFeature,
    width: int,
    height: int,
    frames: int,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """Refuse a priced load inside a windowed job no run has confirmed (1440p, 4K, or a smaller
    canvas that fell back to windows).

    Runs after pricing so the refusal carries the figures and the window the
    price tried, and so a reckless downgrade runs that window. A reject keeps
    its own reason: it already says why the job fails.
    """
    if isinstance(decision, VideoJobReject):
        return decision
    route = budget_route(feature, job_filter.profile)
    windows = job_window_count(frames, decision.chunk_pixel_frames)
    if is_unmeasured_high_res(
        feature, route, width, height, windows=windows
    ) or is_unmeasured_windowed_below_1440p(feature, route, width, height, windows=windows):
        return unmeasured_cell_reject(decision)
    return decision


class PricedLoad(Protocol):
    """What a Fast or IC-LoRA load carries into an unmeasured-cell refusal."""

    @property
    def seq(self) -> float: ...
    @property
    def full_gib(self) -> float: ...
    @property
    def stream_gib(self) -> float: ...
    @property
    def chunk_pixel_frames(self) -> int | None: ...


def unmeasured_cell_reject(load: PricedLoad) -> VideoJobReject:
    return VideoJobReject(
        seq=load.seq,
        full_gib=load.full_gib,
        stream_gib=load.stream_gib,
        reason="unmeasured_cell",
        chunk_pixel_frames=load.chunk_pixel_frames,
    )


def decide_video_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """Fast t2v/i2v/a2v: the priced verdict, refused inside an unmeasured 1440p or 4K cell."""
    return gate_unmeasured_cell(
        _decide_priced_video_job(width, height, frames, job_filter=job_filter),
        feature="fast",
        width=width,
        height=height,
        frames=frames,
        job_filter=job_filter,
    )


def _decide_priced_video_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """Fast t2v/i2v/a2v: advertised (stage-2) pixels, no ×2, no spatial cap.

    A 1440p or 4K clip longer than one window runs in windows and prices one window: the
    longest window that loads, as ``VideoJobLoad.chunk_pixel_frames``. When none loads, the
    verdict is the reject of the shortest window, so the figures name the smallest ask.
    """
    if not is_chunked_canvas(width, height):
        single_pass = _decide_fast_pass(width, height, frames, windows=1, job_filter=job_filter)
        if not _one_pass_cannot_fit_card(single_pass, job_filter):
            return single_pass
        # No bare card of this size runs the clip in one pass, so run it in windows. When no window
        # loads either, the single-pass reject stays the verdict, so nothing else about the
        # refusal (its figures, a reckless downgrade running one pass) changes.
        windowed = _decide_windowed_video_job(width, height, frames, job_filter=job_filter)
        return windowed if isinstance(windowed, VideoJobLoad) else single_pass
    decision: VideoJobDecision | None = None
    if frames <= window_candidates(width, height)[0]:
        decision = _decide_fast_pass(width, height, frames, windows=1, job_filter=job_filter)
        if isinstance(decision, VideoJobLoad):
            return decision
    windowed = _decide_windowed_video_job(width, height, frames, job_filter=job_filter)
    if windowed is not None:
        return windowed
    assert decision is not None  # window_candidates is never empty
    return decision


def _one_pass_cannot_fit_card(single_pass: VideoJobDecision, job_filter: LocalJobFilter) -> bool:
    """A one-pass refusal that no empty card of this size would pass, on a card a run covers.

    Only the GPU refusal, and only when its ask is over the card's total VRAM: a window does not
    shorten a RAM shortfall, a fence or an unsupported process. An ask that fits an empty card is a
    busy or warm card, which frees up: it keeps its refusal, so the evict-and-retry and the
    close-other-apps message stay in charge and a clip that runs in one pass is not windowed.
    """
    if isinstance(single_pass, VideoJobLoad) or single_pass.reason != "stream_over":
        return False
    total = job_filter.budget.total_vram_gib
    return (
        total is not None
        and single_pass.stream_gib > total
        and window_fallback_measured("fast", budget_route("fast", job_filter.profile), total)
    )


def _decide_windowed_video_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
) -> VideoJobDecision | None:
    """The longest window that loads, as ``chunk_pixel_frames``.

    When none loads, the reject of the shortest window, so a reckless downgrade runs that
    window and the figures name the smallest ask. None when the clip is no longer than the
    shortest window, so there is nothing to chunk.
    """
    decision: VideoJobDecision | None = None
    chunk_pixel_frames: int | None = None
    for window in window_candidates(width, height):
        if frames <= window:
            continue  # One pass already covered every window at least this long.
        chunk_pixel_frames = window
        decision = _decide_fast_pass(
            width,
            height,
            window,
            windows=window_count(frames, window),
            job_filter=job_filter,
        )
        if isinstance(decision, VideoJobLoad):
            return replace(decision, chunk_pixel_frames=chunk_pixel_frames)
    if decision is None:
        return None
    return replace(decision, chunk_pixel_frames=chunk_pixel_frames)


def _decide_fast_pass(
    width: int,
    height: int,
    frames: int,
    *,
    windows: int,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """One pass of ``frames`` frames, inside a job of ``windows`` windows."""
    seq = video_tokens(height, width, frames)
    route = budget_route("fast", job_filter.profile)
    if route == "mac_measured":
        return decide_mac_job(
            seq,
            fast_job_stream_gib(seq, windows=windows, darwin=True),
            job_filter=job_filter,
        )
    if route == "windows_measured":
        return _decide_windows_fast_job(seq, windows=windows, job_filter=job_filter)
    return decide_load_mode(
        seq,
        memory_gb=job_filter.budget.memory_gb,
        process_mode=job_filter.profile.process_mode,
        darwin=job_filter.profile.darwin,
    )


@dataclass(frozen=True, slots=True)
class FastDurationOutcome:
    """Per-cell verdicts: kept durations plus why each dropped cell died."""

    kept: list[int]
    rejected: dict[int, VideoJobRejectReason]
    # Full decisions for dropped cells, so the specs layer can message them
    # without re-deciding. Unsupported carries fixed copy (figures ignored).
    rejects: dict[int, VideoJobReject] = field(
        default_factory=dict[int, VideoJobReject]
    )
    # Full decisions for kept cells, so the specs layer can tier them
    # (tight/unmeasured) without re-deciding.
    loads: dict[int, VideoJobLoad] = field(default_factory=dict[int, VideoJobLoad])


def decide_fast_durations(
    width: int,
    height: int,
    fps: int,
    durations: Sequence[int],
    *,
    job_filter: LocalJobFilter,
) -> FastDurationOutcome:
    """Decide every cell, keeping loads and naming each rejection's reason."""
    kept: list[int] = []
    rejected: dict[int, VideoJobRejectReason] = {}
    rejects: dict[int, VideoJobReject] = {}
    loads: dict[int, VideoJobLoad] = {}
    for duration in durations:
        decision = decide_video_job(
            width,
            height,
            compute_num_frames(duration, fps),
            job_filter=job_filter,
        )
        if isinstance(decision, VideoJobLoad):
            kept.append(duration)
            loads[duration] = decision
        else:
            rejected[duration] = decision.reason
            rejects[duration] = decision
    return FastDurationOutcome(
        kept=kept,
        rejected=rejected,
        rejects=rejects,
        loads=loads,
    )


def decide_edit_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """Retake/extend: the priced verdict, refused inside an unmeasured 1440p or 4K cell."""
    return gate_unmeasured_cell(
        _decide_priced_edit_job(width, height, frames, job_filter=job_filter),
        feature="edit",
        width=width,
        height=height,
        frames=frames,
        job_filter=job_filter,
    )


def _decide_priced_edit_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """Retake/extend. Reserved-basis stream law on Windows, with more full-mode headroom.

    The gate prices native cuda_max_reserved (see _EDIT_STREAM): allocated no
    longer gates because an allocated-fitting window paged past physical VRAM.
    Budget the footage processed — extend denoises source frames plus new ones,
    and one token function serves retake and extend (the retake/extend delta is
    denoise length, not the mask).

    Stays out of the Windows Fast profile: that envelope was measured for
    T2V/I2V/A2V, and both the edit floor and the edit law basis differ. On
    Darwin the stream gate is Mac-native (retake == extend, one envelope).
    Off Windows-and-Darwin the legacy generic curve is untouched (it errs
    conservative there). Full-vs-stream keeps the 5.0 headroom: a 5090 that full-loaded 1280x704x337 reserved the whole
    card, and full behavior on current builds is unmeasured either way.
    This gate also enables Mac window-shortening: ``max_edit_frames_that_load``
    trims 1080p windows to the pool instead of approving the full cap.
    """
    seq = video_tokens(height, width, frames)
    route = budget_route("edit", job_filter.profile)
    if route == "windows_measured":
        return decide_measured_job(
            required_gib=_edit_stream_required_gib(seq),
            required_ram_bytes=job_filter.profile.windows_edit_required_ram_bytes,
            seq=seq,
            full_headroom_gib=_EDIT_FULL_HEADROOM_GIB,
            job_filter=job_filter,
        )
    if route == "mac_measured":
        return decide_mac_job(
            seq, _mac_edit_stream_required_gib(seq), job_filter=job_filter
        )
    return decide_load_mode(
        seq,
        memory_gb=job_filter.budget.memory_gb,
        process_mode=job_filter.profile.process_mode,
        darwin=job_filter.profile.darwin,
        full_headroom_gib=_EDIT_FULL_HEADROOM_GIB,
    )


def max_edit_frames_that_load(
    width: int,
    height: int,
    *,
    frame_cap: int,
    job_filter: LocalJobFilter,
) -> int:
    """Largest frame count at or below ``frame_cap`` that ``decide_edit_job`` loads.

    1920x1088x505 asks about 41 GiB on the measured reserved curve and 422s
    everywhere at or under 40 GB.
    That canvas is the letterbox of a 1080p picture. Home passes this cap
    into the encode window so that 1080p job runs at the longest window that
    fits. When nothing fits, return ``frame_cap`` and let the loader 422.
    """
    if frame_cap < 1:
        return frame_cap

    def loads(frames: int) -> bool:
        # Priced, not gated: shortening a window answers memory. The unmeasured-cell gate does not move with length.
        decision = _decide_priced_edit_job(
            width,
            height,
            frames,
            job_filter=job_filter,
        )
        return isinstance(decision, VideoJobLoad)

    if loads(frame_cap):
        return frame_cap
    lo = 1
    hi = frame_cap - 1
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if loads(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best > 0 else frame_cap


def advertised_fast_durations(
    width: int,
    height: int,
    fps: int,
    durations: Sequence[int],
    *,
    job_filter: LocalJobFilter,
) -> list[int]:
    """Keep Fast cells that ``decide_video_job`` would load. Not a full-mode envelope.

    Filtering with ``fits(..., "full_models_loading")`` would hide 720p/20s on a
    5090 that streams it. Darwin never compares free RAM to the CUDA stream curve
    (measured Mac Fast gates on the MPS pool instead).
    """
    # Intentional fast path: unsupported means no list — skip N cell evaluations.
    if job_filter.profile.process_mode == "unsupported":
        return []
    return decide_fast_durations(
        width, height, fps, durations, job_filter=job_filter
    ).kept


# Windows (CUDA) windowed peak above the single-pass law, per token of one window. The
# carry and the earlier windows leave allocator state behind, and that state grows with the
# window's tokens: the peak climbs for two windows past the first, then holds flat for every
# window count measured. A fixed figure fits one window size only: the one fitted at a 4K
# window over-asked the small 41-frame windows, which refuses jobs a 16 GB card loads, and
# under-asked 1440p's 129-frame windows. The coefficients keep a small margin over the 4K
# points; lowering either leaves the 4K cells at zero margin.
_WINDOWED_PLATEAU_GIB_PER_TOKEN = 1.05e-4

# Two windows pay most of the plateau step when the second window is full (1440p at 10 s and
# 24 fps is exactly that: 129 frames then 112 more). A short last window grows much less, and
# the job budget does not see the last window's length, so every two-window job pays the full
# step. That over-asks a short last window (4K at 3 s) until a run measures it.
_WINDOWED_TWO_WINDOW_GIB_PER_TOKEN = 6.5e-5


def _windows_windowed_extra_gib(windows: int, tokens: float) -> float:
    if windows >= PEAK_PLATEAU_WINDOWS:
        return _WINDOWED_PLATEAU_GIB_PER_TOKEN * tokens
    if windows == 2:
        return _WINDOWED_TWO_WINDOW_GIB_PER_TOKEN * tokens
    return 0.0


# Mac (MPS) windowed price above the single-pass law, per token of one window, on the same
# basis as the law itself: the smallest allocator cap that still passes, not the native
# peak, which holds cache the cap would force out. The carry and the earlier windows leave
# allocator state behind that grows with the window's tokens, and the peak climbs for two
# windows past the first and then holds. The Mac price sits above the Windows one because
# the 4K cap runs landed above where the Windows coefficient would put them. A fixed figure
# under-asks the 4K cap edge and the 1440p windows; lowering either coefficient re-opens the
# 4K edge, which the 4K price covers by about 0.1 GiB.
_MAC_WINDOWED_PLATEAU_GIB_PER_TOKEN = 1.30e-4

# Two windows pay most of the plateau step when the second window is full (1440p at 10 s and
# 24 fps). A short last window grows much less (4K at 3 s), and the job budget does not see
# the last window's length, so every two-window job pays the full step. That over-asks the
# 4K 3 s cell by about 3 GiB until a run prices a last window of that length on the Mac.
# The 1440p two-window step is read off native runs, not a capped one.
_MAC_WINDOWED_TWO_WINDOW_GIB_PER_TOKEN = 7.5e-5


def _mac_windowed_extra_gib(windows: int, tokens: float) -> float:
    if windows >= PEAK_PLATEAU_WINDOWS:
        return _MAC_WINDOWED_PLATEAU_GIB_PER_TOKEN * tokens
    if windows == 2:
        return _MAC_WINDOWED_TWO_WINDOW_GIB_PER_TOKEN * tokens
    return 0.0


def fast_job_stream_gib(tokens: float, *, windows: int, darwin: bool) -> float:
    """The stream ask of one Fast pass inside a job of ``windows`` windows. Run logs print it
    next to the measured peak, so a refit has both numbers."""
    if darwin:
        return _mac_fast_stream_required_gib(tokens) + _mac_windowed_extra_gib(windows, tokens)
    return max(
        _windows_fast_stream_required_gib(tokens),
        _windows_fast_reserved_required_gib(tokens) + _windows_windowed_extra_gib(windows, tokens),
    )


def edit_job_stream_gib(tokens: float, *, darwin: bool) -> float:
    """The stream ask of one Retake or Extend window. Run logs print it next to the peak."""
    return _mac_edit_stream_required_gib(tokens) if darwin else _edit_stream_required_gib(tokens)


def _windows_fast_stream_required_gib(tokens: float) -> float:
    """Continuous stream requirement: finite for every resolution, no lookup."""
    return _WINDOWS_FAST_STREAM.required_gib(tokens)


def _windows_fast_reserved_required_gib(tokens: float) -> float:
    """Reserved-basis twin: what the card must back with physical pages, no lookup."""
    return _WINDOWS_FAST_RESERVED_STREAM.required_gib(tokens)


def _windows_fast_full_reserved_required_gib(tokens: float) -> float:
    """Full-mode reserved ask: resident full models plus token growth, no lookup."""
    return _WINDOWS_FAST_FULL_RESERVED.required_gib(tokens)


def _mac_fast_stream_required_gib(tokens: float) -> float:
    """Continuous Mac stream requirement against the MPS pool, no lookup."""
    return _MAC_FAST_STREAM.required_gib(tokens)


def _edit_stream_required_gib(tokens: float) -> float:
    """Edit stream requirement on the reserved-basis row."""
    return _EDIT_STREAM.required_gib(tokens)


def _mac_edit_stream_required_gib(tokens: float) -> float:
    """Mac edit requirement against the MPS pool."""
    return _MAC_EDIT_STREAM.required_gib(tokens)


def _streaming_ram_reject(
    seq: float,
    full_gib: float,
    stream_gib: float,
    *,
    budget: MemoryBudget,
    required_ram_bytes: int,
) -> VideoJobReject | None:
    """Shared streamed-RAM gate: the reject, or None when RAM suffices.

    ram_shortfall on known-insufficient RAM, unsupported when unknown.
    Callers pick the floor (Fast vs interim measured); the verdict shape
    is identical, so it lives here instead of pasted per path.
    """
    match budget.streaming_ram_state(required_bytes=required_ram_bytes):
        case "ok":
            return None
        case "shortfall":
            return VideoJobReject(
                seq=seq,
                full_gib=full_gib,
                stream_gib=stream_gib,
                reason="ram_shortfall",
                required_ram_bytes=required_ram_bytes,
            )
        case "unknown":
            return VideoJobReject(
                seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="unsupported"
            )
        case _ as unreachable:
            assert_never(unreachable)


def _decide_windows_fast_job(
    seq: float,
    *,
    windows: int = 1,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """Windows Fast (2.3 and 2.5) availability from the continuous model, then full-vs-stream pick.

    The stream ask is max(allocated, reserved): allocated predicts OOM
    crashes, reserved predicts WDDM paging, and the gate must refuse both
    failures. Reserved dominates everywhere today (floor 9.90 > 8.00 and a
    steeper line), so it binds; the max keeps both failures gated if a future
    refit ever un-dominates. Evaluates Windows-specific fits (continuous
    requirement against ``min(total, free)`` VRAM; full ask against exact
    total) and runs the shared engine.
    The full ask is max(legacy curve, full-mode reserved law): native full-mode
    reserved exceeds the legacy ask by ~3-4 GiB everywhere measured, so the law
    binds; the max keeps the legacy ask as the floor if a future refit ever
    un-dominates.
    Free can only restrict, never expand, and the full-vs-stream pick stays on
    total for stability. A 16 GB card has ~15.0 GiB free on a working desktop —
    the exact cap that OOM'd after ~150 s in stage-2 denoise, so free-VRAM
    gating is load-bearing. Missing exact bytes with a viable integer figure
    falls back to the integer rather than emptying the grid — floor rounding only
    hides, and only a missing integer too fails closed. The streamed-RAM gate
    reports ``ram_shortfall`` on known-insufficient RAM, ``unsupported`` when
    unknown.
    """
    full_gib = max(
        estimated_giB("full_models_loading", seq),
        _windows_fast_full_reserved_required_gib(seq),
    )
    stream_gib = fast_job_stream_gib(seq, windows=windows, darwin=False)

    available = job_filter.budget.available_vram_gib
    total = job_filter.budget.total_vram_gib
    if available is None or total is None:
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="unsupported"
        )
    decision = decide_from_fits(
        seq,
        full_gib,
        stream_gib,
        stream_fits=(available >= stream_gib),
        full_fits=(full_gib <= _full_memory_gb(total)),
        process_mode=job_filter.profile.process_mode,
        # Windows-only path: profile.darwin is always False here.
        legacy_darwin_never_reject=False,
    )
    if (
        isinstance(decision, VideoJobLoad)
        and decision.mode == "streaming_models_loading"
    ):
        ram_reject = _streaming_ram_reject(
            seq,
            full_gib,
            stream_gib,
            budget=job_filter.budget,
            required_ram_bytes=job_filter.profile.windows_fast_required_ram_bytes,
        )
        if ram_reject is not None:
            return ram_reject
    return decision


# MPS transformer sequence fence: a per-forward token ceiling, not a memory
# curve. Past it the distilled stage-2 forward aborts the process — Metal
# asserts `[MPSNDArrayMatrixMultiplication encodeToCommandBuffer:] destination
# is too large for kernel` and the backend dies with SIGABRT, mid-queue, with
# no catchable error. The failing op is the per-token-timestep AdaLN
# projection: [tokens, 36864]. At 257,040 tokens that is 9.48B elements,
# past the uint32 index range (2^32 = 4.29B); the exact bound at width 36864
# is 116,508 tokens, and the fence sits at 116,000, an epsilon under it.
# AdaLN binds first by construction — at 6-9x the inner width it is the
# widest per-token linear in the arch (FFN is 4x, QKV 3x, patchify 1x), so
# the rest of the family still has headroom at the fence. No extra margin:
# the fence is the hardware's limit, and anything the hardware can run stays
# admitted; an ltx-core bump that widens AdaLN moves the bound and must
# re-derive this constant like any other fitted figure. Bracket: 1080p/10s@24
# (63,240 tokens) passes, 1080p/20s@50fps (257,040 tokens) SIGABRTs. The fence
# binds one forward, never a clip: a windowed Fast job or a tiled IC-LoRA job
# runs one window per forward, so its whole-clip token count (253k at 4K/10s)
# may sit far past the fence while every forward stays under it. The pool
# gate below already refuses most of what the fence admits on small Macs —
# the fence only binds on huge pools that would otherwise crash.
# Every decide_mac_job caller passes the heaviest single forward's tokens
# (Fast: stage-2 canvas; edit: canvas window; IC-LoRA: per-window/per-tile
# pass), so one fence here guards all three features. CUDA has no such
# limit (64-bit indexing), so the fence is Darwin-only by construction.
_MPS_TRANSFORMER_TOKEN_FENCE = 116_000


# Full-resident policy evidence (for decide_mac_job below): interleaved
# DISK-vs-NONE runs. Per-feature bracket evidence lives with each curve
# (_MAC_FAST_STREAM, _MAC_EDIT_STREAM, and the IC-LoRA constants).
def decide_mac_job(
    seq: float,
    stream_gib: float,
    *,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """Darwin availability from the Mac allocator-bracket pool model. Always streams.

    Compares the caller's continuous stream requirement against the sampled
    MPS pool (recommended_max) and runs the shared engine with the legacy
    Darwin branch opted out: that branch never refuses, which was correct
    while no Darwin-native ceiling existed and is exactly what measured Mac
    profiles replace. Full resident stays unselectable (interleaved
    DISK-vs-NONE comparison: full resident only modestly faster at several
    times the pool — never the better pick on shared unified memory).
    Unknown pool fails closed to unsupported. No RAM gate: Mac DISK
    streaming maps from disk with no separate pinned set (that pin is
    CUDA-offload-specific); the pool lives inside total RAM, so the pool
    comparison already covers memory. Process mode flows through, so
    unsupported still rejects first.

    Callers supply the feature's stream requirement; this core owns only the
    pool comparison and the never-full policy. The MPS sequence fence runs
    before the pool comparison (a fence verdict never prints pool figures),
    after unsupported (a dead machine reports unsupported, not the fence).
    """
    full_gib = estimated_giB("full_models_loading", seq)

    if (
        job_filter.profile.process_mode != "unsupported"
        and seq > _MPS_TRANSFORMER_TOKEN_FENCE
    ):
        return VideoJobReject(
            seq=seq,
            full_gib=full_gib,
            stream_gib=stream_gib,
            reason="mps_sequence_over",
        )

    pool = job_filter.budget.mps_pool_gib
    if pool is None:
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="unsupported"
        )
    return decide_from_fits(
        seq,
        full_gib,
        stream_gib,
        stream_fits=(pool >= stream_gib),
        full_fits=False,
        process_mode=job_filter.profile.process_mode,
        legacy_darwin_never_reject=False,
    )


def decide_measured_job(
    *,
    required_gib: float,
    required_ram_bytes: int,
    seq: float,
    full_headroom_gib: float,
    job_filter: LocalJobFilter,
    force_stream: bool = False,
) -> VideoJobDecision:
    """Measured stream requirement + legacy full pick + Windows RAM gate.

    Twin of _decide_windows_fast_job for edit/IC-LoRA on Windows: the stream
    requirement is measured (not the generic curve), compared against
    min(total, free) with the integer fallback; the full-vs-stream pick stays
    on the legacy headroom (full behavior on current builds is unmeasured —
    conservative by design); streamed loads face the
    caller's per-feature RAM floor. Unknown bytes fail closed like Fast. Off
    Windows the callers keep the legacy generic path untouched. ``force_stream``
    skips the full pick for jobs whose full-resident shape is unmeasured (the
    legacy curve on a small seq would full-load cards the true resident
    working set does not fit).
    """
    full_gib = estimated_giB("full_models_loading", seq)
    available = job_filter.budget.available_vram_gib
    total = job_filter.budget.total_vram_gib
    if available is None or total is None:
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=required_gib, reason="unsupported"
        )
    decision = decide_from_fits(
        seq,
        full_gib,
        required_gib,
        stream_fits=(available >= required_gib),
        full_fits=not force_stream
        and fits(
            seq, "full_models_loading", _full_memory_gb(total, full_headroom_gib)
        ),
        process_mode=job_filter.profile.process_mode,
        # Windows-only path: profile.darwin is always False here.
        legacy_darwin_never_reject=False,
    )
    if (
        isinstance(decision, VideoJobLoad)
        and decision.mode == "streaming_models_loading"
        and job_filter.profile.system == "Windows"
    ):
        ram_reject = _streaming_ram_reject(
            seq,
            full_gib,
            required_gib,
            budget=job_filter.budget,
            required_ram_bytes=required_ram_bytes,
        )
        if ram_reject is not None:
            return ram_reject
    return decision
