"""IC-LoRA job budget: heaviest-pass tokens, measured per platform, plus measured spatial No."""


from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from api_types import (
    REFERENCE_STILL_FRAME_IDX,
    DroppedIcLoraDuration,
    IcLoraCatalogItem,
    IcLoraControl,
    IcLoraSettings,
)
from frame_math import compute_num_frames
from runtime_config.fast_chunking import job_window_count
from runtime_config.ic_lora_decode_window import choose_decode_window
from runtime_config.ic_lora_chunking import (
    IC_LORA_CHUNK_PIXEL_FRAMES,
    ic_lora_carry_frames,
    ic_lora_chunk_pixel_frames,
    ic_lora_may_chunk,
)
from runtime_config.ic_lora_local_envelope import (
    IC_LORA_SOURCE_TOO_LARGE,
    IC_LORA_V1_ENVELOPE_MESSAGE,
    effective_denoise_stage1,
    ic_lora_output_canvas,
    over_spatial_cap,
)
from runtime_config.ic_lora_loop_fps import DEFAULT_IC_LORA_FPS, resolve_loop_fps
from runtime_config.ic_lora_stage_mode import IcLoraStageMode, heavy_pass_size
from runtime_config.ic_lora_tiling import IcLoraTiling, transformer_pass_size
from runtime_config.measured_cells import is_unmeasured_high_res
from runtime_config.reject_reasons import IcLoraDroppedReason, JobRiskTier
from runtime_config.video_job_budget import (
    MPS_SEQUENCE_OVER_MESSAGE,
    VIDEO_JOB_REJECT_ERRORS,
    UNMEASURED_CELL_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
    BudgetRoute,
    LoadMode,
    LocalJobFilter,
    VideoJobLoad,
    VideoJobReject,
    budget_route,
    decide_load_mode,
    decide_mac_job,
    decide_measured_job,
    estimated_giB,
    reject_message_and_code,
    risk_for_reject,
    unmeasured_cell_reject,
    untested_readout,
    video_tokens,
)

# Compat-matrix preview of a Home IC-LoRA job. Home runs entries that skip stage 2 and
# entries that keep the IC-LoRA on stage 2 (Layout To Render). This preview is the
# single-stage case, the lightest one. resolution_factor is not a Home control; 1.5 is
# the catalog preview bucket this estimate has always used.
HOME_IC_LORA_PREVIEW_SETTINGS = IcLoraSettings(skip_stage_2=True, resolution_factor=1.5)

# Windows IC-LoRA stream requirement: one reserved line on the heaviest-pass
# token count (``_heaviest_pass_tokens``), every stage layout:
#   required = max(8.00, 6.50 + 2.95e-4 * seq)
# Reserved predicts paging, allocated predicts crashes — the gate thresholds
# predicted paging, so reserved alone binds. A chunked job prices one window, with no
# window-count term. The reserved peak does grow through the first four windows (each pays
# for state the earlier ones left behind) and holds flat after, so the one-window line is an
# upper bound only because the tile line and the canvas term below leave that growth room.
# Floor: streamed resident-model allocation before any denoise. The resident
# model does not care about stage layout.
_IC_LORA_FLOOR_GIB = 8.00
_IC_LORA_RESERVED_INTERCEPT_GIB = 6.50
_IC_LORA_RESERVED_GIB_PER_SEQ_TOKEN = 2.95e-4
# Largest stage-1 token count with a measured two-stage pass. The ceiling is
# the measured maximum; do not extrapolate — an unmeasured large-canvas job
# must refuse rather than price on a line that never saw it. Counted per
# window for a chunked two-stage job.
_IC_LORA_TS_PRICED_MAX_TOKENS = 14_880

# Windows one-window tiled cell: a tiled job prices its tile on the Windows reserved
# line plus a 1.30 GiB canvas decode term, inside the measured cell only (single-stage,
# unchunked, at most 97 frames, tile-pass token count at most 14,976 with stills
# counting). The decode term pays the VAE decode, which runs on the canvas while
# denoise runs on the tile. Two one-window runs show the gap: Restore peaks 1.28 GiB
# over its tile ask, Refine 0.79 GiB over — so the term covers the worse of the two.
# Small grids skip the term: the single-tile run (R1, 960x512 canvas) peaks at
# 7.96 GiB on denoise alone, under its 10.18 GiB ask, so at or under that grid the
# peak stays under the ask with no term. Chunked+tiled stays canvas-priced: per-window
# reserved peaks grow as streamed decode accumulates (8.54, 11.71, then 13.38 GiB on
# the first three windows), then saturate — the 7-window follow-up run repeats those
# three peaks exactly, then holds around 14.5 GiB from window 4 on, under its
# 22.15 GiB canvas ask. So one window is not the whole ask. Tiled always streams:
# full-resident tiled is unmeasured, and the legacy full curve on a tile token count
# would full-load 32 GB cards, where resident weights (~23 GiB) plus the tile working
# set exceed the headroom.
# High-resolution tiled cell (1440p and 2160p, any catalog LoRA): ``high_res_tiling``
# raises the spatial cap, so the tile is priced, not the canvas. The tile leaves out the
# canvas-sized latent, the reference encode and the streamed decode, so a term per
# megapixel of canvas pays for them. The term was scaled from the 1080p Restore run
# (2.09 Mpx), which peaked 14.58 GiB, and the 1.75 GiB per Mpx came from the Refine tile
# ask (10.92 GiB); Restore's own tile ask is 10.41, which would give 2.00. The smaller
# figure stays: a 25% margin (2.2) priced a 4K canvas with a 1920x1088 tile at 32.1 GiB and
# refused it on a 32 GB card. One Windows 4K run now exists and peaked 22.77 GiB under a
# 28.34 ask, so on Windows the term over-asks at 4K by about 5.6 GiB, which costs no
# admission on a 32 GB card. Windows 1440p has no run, so its price is interpolated.
# The Mac adds the same term to its flat pass above 1080p, so the Mac inherits a Windows
# number: no Mac high-res tiled run exists at any canvas, which is why Standard mode
# refuses those cells as unmeasured. Tightening the term needs a 1440p run, and pricing the
# Mac needs its own 4K run.
_HIGH_RES_TILED_GIB_PER_MPX = 1.75
_HIGH_RES_TILED_BASE_MPX = 1920 * 1088 / 1e6
_WINDOWS_TILED_MAX_TILE_SEQ = 14_976
_WINDOWS_TILED_MAX_FRAMES = 97
_WINDOWS_TILED_SINGLE_PX = 960 * 512
_WINDOWS_TILED_DECODE_GIB = 1.30


def _windows_iclora_required_gib(seq: float) -> float:
    return max(
        _IC_LORA_FLOOR_GIB,
        _IC_LORA_RESERVED_INTERCEPT_GIB + _IC_LORA_RESERVED_GIB_PER_SEQ_TOKEN * seq,
    )


# Mac (Darwin) IC-LoRA stream requirement, Mac-native fit (no transfer from CUDA):
# max(9.84, 5.75 + 6.105e-4 * denoise_tokens), +0.50 for ingredients.
# Unified memory costs more per token than discrete VRAM: the allocator working
# set is larger and a per-token transient folds into the slope. The floor binds
# small jobs — the slope starts inside landscape sizes, so it is not dead code.
# Other adapters gate on the same curve; only ingredients prices the adder below.
# A mid-range interpolation check sits on the fail cap: no refit on one wide
# point — refit on field OOMs near a pool boundary or a dedicated bracket.
#
# Ingredients adapter term, flat: measured tops sit a token-flat amount over the
# fitted video line, so the term is a flat adder, not a slope.
# Keyed on the adapter id, never the feed: image-fed jobs already sit under the
# line, so a blanket image adder would stack on an over-covered cell.
# Keyed on identity, never weight size: a same-size adapter passes on the line.
# Do not extend this to another adapter, or key it on feed/size, without a
# bracket above the line; what would replace it is a per-adapter table once a
# second heavy adapter brackets above the line.
_MAC_IC_LORA_FLOOR_GIB = 9.84
_MAC_IC_LORA_INTERCEPT_GIB = 5.75
_MAC_IC_LORA_GIB_PER_TOKEN = 6.105e-4
_MAC_IC_LORA_INGREDIENTS_ADDER_GIB = 0.50
# Public scalar for matrix prose. Single source: the row above.
MAC_IC_LORA_FLOOR_GIB = _MAC_IC_LORA_FLOOR_GIB


def _mac_iclora_stream_required_gib(
    denoise_tokens: float, ic_lora_id: str | None
) -> float:
    """Single-stage Mac requirement plus the ingredients-keyed adder.

    ``max(_MAC_IC_LORA_FLOOR_GIB, _MAC_IC_LORA_INTERCEPT_GIB +
    _MAC_IC_LORA_GIB_PER_TOKEN * denoise_tokens)``, with
    ``_MAC_IC_LORA_INGREDIENTS_ADDER_GIB`` added outside the max when the
    adapter id is ingredients. Every other id — and the id-less custom
    path — prices on the line.
    """
    required = max(
        _MAC_IC_LORA_FLOOR_GIB,
        _MAC_IC_LORA_INTERCEPT_GIB + _MAC_IC_LORA_GIB_PER_TOKEN * denoise_tokens,
    )
    if ic_lora_id == "ingredients":
        required += _MAC_IC_LORA_INGREDIENTS_ADDER_GIB
    return required

# Mac two-stage requirement on the heaviest-pass count. Both two-stage layouts price
# on it: the opt-in probe (O1) lands 0.17 GiB above the chord ask at 13,888 tokens,
# so cost follows that count across layouts:
#   required = max(9.84, 9.25 + 1.861e-4 * seq, 6.20 + 3.0303e-4 * seq)
# Two chords because one line cannot hold both frame sizes: the 540p chord
# under-asks the 720p 10-second pass bottom by ~3.4 GiB, so the price pays the higher
# of the 540p chord and the chord through the 720p pass bottoms. The 1080p pass
# bottom lands under the higher chord, so 1080p prices on it — a third line
# would only add slack. Fenced to the measured cell, not just the lines:
# untiled, stills <= 1, stage-1 at most 928x512 px, and per-window at most 7,424
# tokens and 241 frames. The canvas fence matters because a small-canvas short job
# slides under the token fence; the duration fence matters because a 540p 15-second
# unchunked job slides under both — and the chords under-ask it by about 3 GiB there,
# so the fence carries real money, not just caution. Chunked two-stage (Layout to
# Render) prices flat in windows on the per-window chord: K (7w) and K4 (4w) both
# edge (12, 14] GiB against a 13.67 GiB window ask, and H10c (1080p 10s, 3w) edges
# (21, 23] GiB against a 21.39 GiB window ask. Anything past the fences refuses
# with mac_two_stage_unmeasured.
_MAC_L2R_PRICED_MAX_PX = 928 * 512
_MAC_L2R_PRICED_MAX_TOKENS = 7_424
_MAC_L2R_PRICED_MAX_FRAMES = 241
_MAC_L2R_INTERCEPT_GIB = 9.25
_MAC_L2R_GIB_PER_SEQ_TOKEN = 1.861e-4
_MAC_L2R_720P_INTERCEPT_GIB = 6.20
_MAC_L2R_720P_GIB_PER_SEQ_TOKEN = 3.0303e-4


def _mac_iclora_two_stage_required_gib(seq: float) -> float:
    """Mac two-stage requirement on the heaviest-pass count.

    Only called for jobs ``_mac_two_stage_priced`` admits; the fences there own
    the measured cell, these lines own the ask. Both two-stage layouts read the
    same chords — the opt-in probe priced on them, so the function does not take
    a layout.
    """
    return max(
        _MAC_IC_LORA_FLOOR_GIB,
        _MAC_L2R_INTERCEPT_GIB + _MAC_L2R_GIB_PER_SEQ_TOKEN * seq,
        _MAC_L2R_720P_INTERCEPT_GIB + _MAC_L2R_720P_GIB_PER_SEQ_TOKEN * seq,
    )

# Mac tiled cell: a tiled Mac job prices its tile only inside the measured
# one-window cell (single-stage, unchunked, tile-pass seq <= 13,260 with stills
# counting, window frames <= 97, canvas <= 1920x1088). The tile leaves out the
# full-canvas latent, VAE and reference encode, so outside the cell the job
# prices at the canvas — the safe direction — except big-canvas chunked restores
# below, which price the flat S20 pass.
_MAC_TILED_PRICED_MAX_SEQ = 13_260
_MAC_TILED_PRICED_MAX_FRAMES = 97
_MAC_TILED_PRICED_MAX_PX = 1920 * 1088

# Mac chunked+tiled flat price: 18.7 for a big-canvas chunked tiled job through
# 20 s (single-stage, chunked, canvas past 720p, clip <= 481 frames, tile-pass
# seq at the window <= 13,260 with stills counting). Chunked+tiled breaks the
# tile price — the 20 s clip fails at the 9.84 tile cap and passes at 18.7 — so
# one window is not the ask and the clip prices the pass, flat: fewer windows
# need no more than the measured seven, so the 20 s pass safely covers 10 s too.
# Small canvases keep canvas-window pricing (their growth fits inside it); past
# 481 frames growth is unmeasured and canvas-window pricing stands on its slack.
_MAC_CHUNKED_TILED_FLAT_GIB = 18.7
_MAC_CHUNKED_TILED_FENCE_PX = 1280 * 704
_MAC_CHUNKED_TILED_MAX_FRAMES = 481
_MAC_CHUNKED_TILED_MAX_TILE_SEQ = 13_260

# Full-vs-stream pick stays conservative: 8s+ streams on 32 GB. Forced-full
# runs pass clean, yet streaming runs as fast or faster — so the conservative
# pick costs nothing while a full boundary stays unmeasured. The headroom is a
# miss by design (6 would full-load 8s, 8 would stream 5s); re-measure the full
# boundary rather than tuning it.
_IC_LORA_FULL_HEADROOM_GIB = 7.0

# Two-stage is offered where it is measured and refused past the token ceiling.
# A token rule, not a duration rule: a long small-canvas job can price while an
# unmeasured large-canvas short job must not slip under a clock. It bites opt-in
# two-stage and Layout to Render; single-stage never reaches it.
TWO_STAGE_OVER_MEASURED_MESSAGE = (
    "This two-stage clip is longer than we have measured at this photo size. "
    "Try a shorter clip, a smaller photo, or single-stage for the full length."
)
# Mac two-stage past the measured cell refuses, not fenced by the Windows token
# range: both two-stage layouts inside the L2R cell are priced
# (``_mac_two_stage_priced`` owns that cell), so past-1080p, tiled, and
# long unchunked small-canvas still refuse here rather than price on chords that never saw
# them. The picker keeps the durations selectable with this refusal attached;
# generate/preflight 422 with it.
MAC_TWO_STAGE_UNMEASURED_MESSAGE = (
    "Two-stage upscaling isn't measured on Macs yet, so it stays off here. "
    "Use single-stage for the full length."
)
# Long unchunked tiled Refine past 720p refuses on every route: the DiffVAE decode
# cannot fit its minimum tile (stage 4 needs 17.2 GiB vs 11.7 usable at
# 1920x1088x241) and the pipeline answers 400 before denoise starts. Canvas pricing
# would admit this shape on 48 GB cards straight into that 400, so the shape refuses
# rather than prices. At or under 720p the decode working set fits usable with margin
# and canvas pricing (which only admits big cards) stands.
# The same copy answers the decode-window check (``choose_decode_window``): any tiled high-res
# LoRA (Refine Details, Restore, Day to Night) can hit it, so it names no LoRA.
REFINE_DECODE_FLOOR_MESSAGE = (
    "This high-resolution clip is too large for this GPU's decoder to finish. "
    "Try a shorter clip or a lower resolution."
)
_REFINE_DECODE_FENCE_PX = 1280 * 704
_REFINE_DECODE_FENCE_FRAMES = 97


def two_stage_exceeds_priced(
    *, stage_mode: IcLoraStageMode, stage1_width: int, stage1_height: int, frames: int
) -> bool:
    """True when a two-stage job runs past the priced token range.

    Single source for the rule: decide_ic_lora_job calls it as its first
    pre-check, with the window length for a chunked job. A token ceiling needs no
    frame-rate grid, so there is no fps argument at all.
    """
    if stage_mode is IcLoraStageMode.SINGLE_STAGE:
        return False
    return (
        video_tokens(stage1_height, stage1_width, frames)
        > _IC_LORA_TS_PRICED_MAX_TOKENS
    )


def refine_decode_floor_exceeded(
    *,
    stage1_width: int,
    stage1_height: int,
    window: int | None,
    window_frames: int,
    tiling: IcLoraTiling | None,
) -> bool:
    """True when a tiled job runs unchunked past the decodable shape.

    The DiffVAE floor is a pipeline limit, not a budget curve, so the fence is
    platform-blind like the spatial cap. A fixed-window LoRA (Restore) always chunks
    past its window and never reaches this shape; a never-chunking one (Refine)
    does. Counted per window like the token fence, though only unchunked jobs
    can trip it.
    """
    if tiling is None:
        return False
    if window is not None:
        return False
    if window_frames <= _REFINE_DECODE_FENCE_FRAMES:
        return False
    return stage1_width * stage1_height > _REFINE_DECODE_FENCE_PX


# List-time duration filter has no user image. skip_stage_2: 9:16 768×1280 through
# the 768 bucket × factor 1.5 is stage-1 576×960 (Ingredients default). Two-stage
# sizes to the user's source, so preview a local 1080p sheet or the filter is
# optimistic (~8 GiB under a real 1080p two-stage 20s).
_CATALOG_PREVIEW_SKIP_STAGE2_WIDTH = 768
_CATALOG_PREVIEW_SKIP_STAGE2_HEIGHT = 1280
_CATALOG_PREVIEW_TWO_STAGE_WIDTH = 1920
_CATALOG_PREVIEW_TWO_STAGE_HEIGHT = 1088

@dataclass(frozen=True, slots=True)
class IcLoraJobLoad:
    mode: LoadMode
    seq: float
    full_gib: float
    stream_gib: float
    # Window length when the job runs chunked, else None. The pipeline runs this value.
    chunk_pixel_frames: int | None = None


@dataclass(frozen=True, slots=True)
class IcLoraJobReject:
    reason: Literal["spatial", "unsupported", "two_stage_over_measured", "mac_two_stage_unmeasured", "refine_decode_floor"]
    seq: float
    full_gib: float
    stream_gib: float
    # Uniform with VideoJobReject; always None (these verdicts never compare
    # RAM — spatial is a canvas fence, unsupported short-circuits first, and
    # the token-ceiling, Mac-two-stage and decode-floor fences refuse before
    # any curve runs).
    required_ram_bytes: int | None = None
    # Window ``price()`` tried. A reckless downgrade runs this window;
    # None would be one unchunked pass.
    chunk_pixel_frames: int | None = None


IcLoraJobDecision = IcLoraJobLoad | IcLoraJobReject | VideoJobReject


def ic_lora_sequence_tokens(width: int, height: int, frames: int) -> float:
    """Video tokens plus the appended reference clip (same H×W×frames)."""
    return video_tokens(height, width, frames) * 2.0


def _heaviest_pass_tokens(
    stage_mode: IcLoraStageMode,
    stage1_width: int,
    stage1_height: int,
    frames: int,
    stills: int,
    tiling: IcLoraTiling | None,
) -> float:
    """Tokens of the heaviest pass of the job.

    Stage 2 of a STAGE_1_LORA job has no reference, but it runs at twice the stage-1
    size. That is four times the stage-1 tokens, so it is heavier than stage 1 with its
    reference (two times). The other layouts carry the reference in their largest pass.
    Each reference still adds one latent frame of tokens to that pass. A STAGE_1_LORA job
    takes no still.

    A tiled job (single stage) runs the transformer on one tile at a time. Its pass is the
    tile, with the reference cropped to the same tile.
    """
    width, height = transformer_pass_size(*heavy_pass_size(stage_mode, stage1_width, stage1_height), tiling)
    if stage_mode is IcLoraStageMode.STAGE_1_LORA:
        return max(
            video_tokens(height, width, frames),
            ic_lora_sequence_tokens(stage1_width, stage1_height, frames),
        )
    still_tokens = stills * (height / 32.0) * (width / 32.0)
    return ic_lora_sequence_tokens(width, height, frames) + still_tokens


def _mac_two_stage_priced(
    *,
    route: BudgetRoute,
    stage_mode: IcLoraStageMode,
    stage1_width: int,
    stage1_height: int,
    window_frames: int,
    stills: int,
    tiling: IcLoraTiling | None,
) -> bool:
    """True when a Mac two-stage job sits inside the measured cell.

    The chords price both two-stage layouts through the L2R cell (unchunked or
    chunked per window), untiled, with at most the measured look still. Anything
    else two-stage on Mac is unmeasured and refuses. Single-stage never reaches
    this predicate (the call site only invokes it past the single-stage check),
    but the guard stays so a mis-called single-stage job refuses rather than
    prices on the wrong curve.
    """
    if route != "mac_measured":
        return False
    if stage_mode is IcLoraStageMode.SINGLE_STAGE:
        return False
    if tiling is not None:
        return False
    if stills > 1:
        return False
    if window_frames > _MAC_L2R_PRICED_MAX_FRAMES:
        return False
    if stage1_width * stage1_height > _MAC_L2R_PRICED_MAX_PX:
        return False
    return (
        video_tokens(stage1_height, stage1_width, window_frames)
        <= _MAC_L2R_PRICED_MAX_TOKENS
    )


def _mac_tile_priced(
    *,
    route: BudgetRoute,
    stage_mode: IcLoraStageMode,
    stage1_width: int,
    stage1_height: int,
    window: int | None,
    window_frames: int,
    stills: int,
    tiling: IcLoraTiling | None,
) -> bool:
    """True when a Mac tiled job sits inside the measured one-window tile cell.

    The tile-pass seq fence counts stills (a still pushes a tile at the fence
    back to canvas pricing). Anything outside the cell prices at the canvas,
    except a big-canvas chunked restore, which prices the flat S20 pass
    (``_mac_chunked_tiled_flat``).
    """
    if route != "mac_measured":
        return False
    if stage_mode is not IcLoraStageMode.SINGLE_STAGE:
        return False
    if tiling is None:
        return False
    if window is not None:
        return False
    if window_frames > _MAC_TILED_PRICED_MAX_FRAMES:
        return False
    if stage1_width * stage1_height > _MAC_TILED_PRICED_MAX_PX:
        return False
    return (
        _heaviest_pass_tokens(
            stage_mode, stage1_width, stage1_height, window_frames, stills, tiling
        )
        <= _MAC_TILED_PRICED_MAX_SEQ
    )


def _mac_chunked_tiled_flat(
    *,
    route: BudgetRoute,
    stage_mode: IcLoraStageMode,
    stage1_width: int,
    stage1_height: int,
    frames: int,
    window: int | None,
    window_frames: int,
    stills: int,
    tiling: IcLoraTiling | None,
) -> bool:
    """True when a Mac chunked tiled job prices the flat S20 pass.

    The 20 s pass covers shorter chunked clips (fewer windows accumulate less),
    so the 10 s shape prices it without its own run. The tile-pass seq fence
    counts stills at the window. Small canvases and past-20 s clips keep
    canvas-window pricing.
    """
    if route != "mac_measured":
        return False
    if stage_mode is not IcLoraStageMode.SINGLE_STAGE:
        return False
    if tiling is None:
        return False
    if window is None:
        return False
    if frames > _MAC_CHUNKED_TILED_MAX_FRAMES:
        return False
    if stage1_width * stage1_height <= _MAC_CHUNKED_TILED_FENCE_PX:
        return False
    return (
        _heaviest_pass_tokens(
            stage_mode, stage1_width, stage1_height, window_frames, stills, tiling
        )
        <= _MAC_CHUNKED_TILED_MAX_TILE_SEQ
    )


def _windows_tiled_priced(
    *,
    route: BudgetRoute,
    stage_mode: IcLoraStageMode,
    stage1_width: int,
    stage1_height: int,
    window: int | None,
    window_frames: int,
    stills: int,
    tiling: IcLoraTiling | None,
) -> bool:
    """True when a Windows tiled job sits inside the measured one-window tile cell.

    The tile-pass seq fence counts stills (a still pushes a tile at the fence
    back to canvas pricing). Anything outside the cell prices at the canvas.
    """
    if route != "windows_measured":
        return False
    if stage_mode is not IcLoraStageMode.SINGLE_STAGE:
        return False
    if tiling is None:
        return False
    if window is not None:
        return False
    if window_frames > _WINDOWS_TILED_MAX_FRAMES:
        return False
    return (
        _heaviest_pass_tokens(
            stage_mode, stage1_width, stage1_height, window_frames, stills, tiling
        )
        <= _WINDOWS_TILED_MAX_TILE_SEQ
    )


def _is_high_res_tiled(route: BudgetRoute, stage_mode: IcLoraStageMode, tiling: IcLoraTiling | None) -> bool:
    """True for a measured-route job that runs a high cell as tiles (``high_res_tiling``)."""
    return (
        route != "legacy"
        and stage_mode is IcLoraStageMode.SINGLE_STAGE
        and tiling is not None
        and tiling.output_cap() is not None
    )


def decide_ic_lora_job(
    stage1_width: int,
    stage1_height: int,
    frames: int,
    *,
    stage_mode: IcLoraStageMode,
    job_filter: LocalJobFilter,
    ic_lora_id: str | None = None,
    stills: int = 0,
    tiling: IcLoraTiling | None = None,
) -> IcLoraJobDecision:
    """422 definite-over jobs; per-job stream when full will not fit; the chunk window.

    ``stage_mode`` sets the heaviest pass (``heavy_pass_size``). That pass sets the priced
    token count, the chunk rule and the spatial cap. The spatial cap checks the output
    size, so a two-stage job is capped at its stage-2 canvas, not at stage 1. It is
    required (no default) so no call site can silently price a two-stage job as
    single-stage.

    This is the one place that decides chunking. A job chunks when the size and token
    rule says so, or when it may chunk and does not fit unchunked.
    ``IcLoraJobLoad.chunk_pixel_frames`` carries the decision to the pipeline. A chunked
    job holds one window at a time, so it is priced (and fenced) as one window.

    ``stills`` is the number of reference stills. Each adds one latent frame of tokens to
    each window, so a chunked job pays for it once per window.

    ``tiling`` is the LoRA's tile size. It always drives the chunk rule, because the
    pipeline tiles. It sizes the priced pass on the legacy route, on Windows inside
    the measured one-window tile cell (``_windows_tiled_priced``, tile ask plus a
    canvas decode term — the decode runs on the canvas while denoise runs on the
    tile), and on Mac inside its own one-window cell (``_mac_tile_priced``):
    anywhere else a tiled run is unmeasured, and the tile leaves out the
    full-canvas latent, VAE and reference encode, so the job prices as untiled
    at its canvas — except a big-canvas chunked Mac restore, which prices the
    flat S20 pass (``_mac_chunked_tiled_flat``). Unchunked tiled past one window
    still pays the decode term on Windows (its decode volume exceeds anything
    measured); long unchunked Refine past 720p refuses outright at the decode
    floor (``refine_decode_floor_exceeded``).

    Per window, in order: unsupported, the spatial cap, the two-stage token fence, the
    Mac two-stage refusal (unmeasured two-stage shapes only — both two-stage
    layouts inside the L2R cell price on its chords), then the platform curve. Windows prices the measured
    reserved line (``_windows_iclora_required_gib``) against min(total, free) with the
    feature RAM floor. Darwin prices the Mac-native curves (single-stage on
    stage-1 tokens, priced two-stage on the heaviest pass), with the
    ingredients adder keyed on ``ic_lora_id``. Everything else keeps the legacy generic
    curve. The full-vs-stream pick stays conservative (full fits clean but runs
    no faster, so streaming 8s+ costs nothing).
    """
    token_width, token_height = heavy_pass_size(stage_mode, stage1_width, stage1_height)
    route = budget_route("iclora", job_filter.profile)
    # Set below, once the window is chosen: no window fits the decode of the job's last window.
    decode_no_fit = False

    def price(window: int | None) -> IcLoraJobDecision:
        def stamped(decision: IcLoraJobDecision) -> IcLoraJobDecision:
            # Loads already carry the window. Rejects must too: a reckless
            # downgrade runs this window, and None is one unchunked pass.
            if isinstance(decision, IcLoraJobLoad):
                return decision
            return replace(decision, chunk_pixel_frames=window)
        window_frames = frames if window is None else window
        windows_tiled = _windows_tiled_priced(
            route=route,
            stage_mode=stage_mode,
            stage1_width=stage1_width,
            stage1_height=stage1_height,
            window=window,
            window_frames=window_frames,
            stills=stills,
            tiling=tiling,
        )
        high_res_tiled = _is_high_res_tiled(route, stage_mode, tiling)
        canvas_mpx = stage1_width * stage1_height / 1e6
        priced_tiling = (
            tiling
            if route == "legacy"
            or windows_tiled
            or high_res_tiled
            or _mac_tile_priced(
                route=route,
                stage_mode=stage_mode,
                stage1_width=stage1_width,
                stage1_height=stage1_height,
                window=window,
                window_frames=window_frames,
                stills=stills,
                tiling=tiling,
            )
            else None
        )
        seq = _heaviest_pass_tokens(
            stage_mode, stage1_width, stage1_height, window_frames, stills, priced_tiling
        )
        full_gib = estimated_giB("full_models_loading", seq)
        stream_gib = estimated_giB("streaming_models_loading", seq)

        if job_filter.profile.process_mode == "unsupported":
            return stamped(IcLoraJobReject(reason="unsupported", seq=seq, full_gib=full_gib, stream_gib=stream_gib))
        # Spatial before the fences: no clip length passes an over-cap canvas, so
        # "try a shorter clip" would be the wrong advice.
        if over_spatial_cap(token_width, token_height, tiling):
            return stamped(IcLoraJobReject(reason="spatial", seq=seq, full_gib=full_gib, stream_gib=stream_gib))
        if two_stage_exceeds_priced(
            stage_mode=stage_mode,
            stage1_width=stage1_width,
            stage1_height=stage1_height,
            frames=window_frames,
        ):
            return stamped(IcLoraJobReject(
                reason="two_stage_over_measured", seq=seq, full_gib=full_gib, stream_gib=stream_gib
            ))
        if refine_decode_floor_exceeded(
            stage1_width=stage1_width,
            stage1_height=stage1_height,
            window=window,
            window_frames=window_frames,
            tiling=tiling,
        ):
            return stamped(IcLoraJobReject(
                reason="refine_decode_floor", seq=seq, full_gib=full_gib, stream_gib=stream_gib
            ))
        if decode_no_fit:
            return stamped(IcLoraJobReject(
                reason="refine_decode_floor", seq=seq, full_gib=full_gib, stream_gib=stream_gib
            ))
        if (
            stage_mode is not IcLoraStageMode.SINGLE_STAGE
            and route == "mac_measured"
            and not _mac_two_stage_priced(
                route=route,
                stage_mode=stage_mode,
                stage1_width=stage1_width,
                stage1_height=stage1_height,
                window_frames=window_frames,
                stills=stills,
                tiling=tiling,
            )
        ):
            return stamped(IcLoraJobReject(
                reason="mac_two_stage_unmeasured", seq=seq, full_gib=full_gib, stream_gib=stream_gib
            ))

        if route == "windows_measured":
            required_gib = _windows_iclora_required_gib(seq)
            if high_res_tiled:
                required_gib += _HIGH_RES_TILED_GIB_PER_MPX * canvas_mpx
            if windows_tiled and stage1_width * stage1_height > _WINDOWS_TILED_SINGLE_PX:
                required_gib += _WINDOWS_TILED_DECODE_GIB
            if (
                tiling is not None
                and window is None
                and window_frames > _WINDOWS_TILED_MAX_FRAMES
            ):
                required_gib += _WINDOWS_TILED_DECODE_GIB
            decision: VideoJobLoad | VideoJobReject = decide_measured_job(
                required_gib=required_gib,
                required_ram_bytes=job_filter.profile.windows_iclora_required_ram_bytes,
                seq=seq,
                full_headroom_gib=_IC_LORA_FULL_HEADROOM_GIB,
                job_filter=job_filter,
                force_stream=windows_tiled or high_res_tiled,
            )
        elif route == "mac_measured":
            # The single-stage curve was fitted on stage-1 tokens; the heaviest
            # pass of a single-stage job is twice that. The two-stage chord was
            # fitted on the heaviest pass directly; the refusal above fenced out
            # everything past its cell, so only priced two-stage reaches here.
            if high_res_tiled:
                # Past the measured 1080p flat pass, the canvas term grows with the pixels.
                stream_gib = _MAC_CHUNKED_TILED_FLAT_GIB + _HIGH_RES_TILED_GIB_PER_MPX * max(
                    0.0, canvas_mpx - _HIGH_RES_TILED_BASE_MPX
                )
            elif stage_mode is IcLoraStageMode.SINGLE_STAGE:
                if _mac_chunked_tiled_flat(
                    route=route,
                    stage_mode=stage_mode,
                    stage1_width=stage1_width,
                    stage1_height=stage1_height,
                    frames=frames,
                    window=window,
                    window_frames=window_frames,
                    stills=stills,
                    tiling=tiling,
                ):
                    stream_gib = _MAC_CHUNKED_TILED_FLAT_GIB
                else:
                    stream_gib = _mac_iclora_stream_required_gib(seq / 2.0, ic_lora_id)
            else:
                stream_gib = _mac_iclora_two_stage_required_gib(seq)
            decision = decide_mac_job(
                seq,
                stream_gib,
                job_filter=job_filter,
            )
        else:
            decision = decide_load_mode(
                seq,
                memory_gb=job_filter.budget.memory_gb,
                process_mode=job_filter.profile.process_mode,
                darwin=job_filter.profile.darwin,
                full_headroom_gib=_IC_LORA_FULL_HEADROOM_GIB,
            )
        if isinstance(decision, VideoJobReject):
            return stamped(decision)
        return IcLoraJobLoad(
            mode=decision.mode,
            seq=decision.seq,
            full_gib=decision.full_gib,
            stream_gib=decision.stream_gib,
            chunk_pixel_frames=window,
        )

    window = ic_lora_chunk_pixel_frames(
        stage1_width, stage1_height, frames, stage_mode=stage_mode, tiling=tiling
    )
    total_vram_gib = job_filter.budget.total_vram_gib
    if route == "windows_measured" and _is_high_res_tiled(route, stage_mode, tiling) and total_vram_gib is not None:
        # The decode of the job's last window must fit, or the pipeline refuses after the denoise.
        decode_choice = choose_decode_window(
            frames=frames,
            window=window,
            width=stage1_width,
            height=stage1_height,
            total_vram_gib=total_vram_gib,
            carry_frames=ic_lora_carry_frames(tiling),
        )
        window = decode_choice.window
        decode_no_fit = not decode_choice.fits

    def gated(verdict: IcLoraJobDecision) -> IcLoraJobDecision:
        if isinstance(verdict, IcLoraJobLoad) and is_unmeasured_high_res(
            "iclora",
            route,
            token_width,
            token_height,
            windows=job_window_count(
                frames, verdict.chunk_pixel_frames, ic_lora_carry_frames(tiling)
            ),
            ic_lora_id=ic_lora_id,
        ):
            return unmeasured_cell_reject(verdict)
        return verdict

    decision = price(window)
    if (
        window is None
        and isinstance(decision, VideoJobReject)
        and decision.reason == "stream_over"
        and ic_lora_may_chunk(stage1_width, stage1_height, frames, stage_mode=stage_mode, tiling=tiling)
    ):
        # Without this, a clip just under the token rule is priced whole and a longer
        # clip is priced as one window, so the longer clip would be cheaper.
        return gated(price(IC_LORA_CHUNK_PIXEL_FRAMES))
    return gated(decision)


def catalog_preview_stage1(settings: IcLoraSettings) -> tuple[int, int]:
    """Stage-1 pixels for list-time duration filtering (no user image yet)."""
    if settings.skip_stage_2:
        input_width, input_height = (
            _CATALOG_PREVIEW_SKIP_STAGE2_WIDTH,
            _CATALOG_PREVIEW_SKIP_STAGE2_HEIGHT,
        )
    else:
        input_width, input_height = (
            _CATALOG_PREVIEW_TWO_STAGE_WIDTH,
            _CATALOG_PREVIEW_TWO_STAGE_HEIGHT,
        )
    canvas_w, canvas_h, factor = ic_lora_output_canvas(
        skip_stage_2=settings.skip_stage_2,
        resolution_factor=settings.resolution_factor,
        input_width=input_width,
        input_height=input_height,
        resolution=None,
        stage_2_ic_lora=settings.stage_2_ic_lora,
    )
    return effective_denoise_stage1(
        canvas_w, canvas_h, skip_stage_2=settings.skip_stage_2, resolution_factor=factor
    )


def catalog_preview_stills(item: IcLoraCatalogItem) -> int:
    """Reference stills the list-time preview prices: the optional look still counts as
    present, so the filter is not optimistic."""
    return int(item.allows_reference_image and item.reference_image_frame == REFERENCE_STILL_FRAME_IDX)


# Reasons with fixed copy: never print a compared figure for these. The
# message router (dropped_duration_message_and_code) owns this set; a new
# fixed-copy reason must add a branch there, or it falls through to the
# shared builder and prints figures it never compared.
_FIXED_COPY_REASONS: frozenset[IcLoraDroppedReason] = frozenset(
    {"spatial", "unsupported", "two_stage_over_measured", "mac_two_stage_unmeasured", "refine_decode_floor", "mps_sequence_over"}
)


def iclora_drop_risk_tier(reason: IcLoraDroppedReason) -> JobRiskTier | None:
    """Risk tier for an IC-LoRA duration drop. None when a hard fence or
    unsupported: those never allow-with-warning in any mode."""
    if reason == "stream_over" or reason == "ram_shortfall":
        return "over"
    if reason in ("two_stage_over_measured", "mac_two_stage_unmeasured", "unmeasured_cell"):
        return "unmeasured"
    # spatial, refine_decode_floor, unsupported, mps_sequence_over
    return None


@dataclass(frozen=True, slots=True)
class IcLoraDurationOutcome:
    """Per-duration verdicts: kept plus why each dropped duration died."""

    kept: list[int]
    rejected: dict[int, IcLoraDroppedReason]
    # Full decisions for dropped durations, so list-time drops message without
    # re-deciding. Fixed-copy verdicts (spatial, fences, unsupported) carry
    # fixed messages instead — never print a number the budget did not compare.
    rejects: dict[int, VideoJobReject | IcLoraJobReject] = field(
        default_factory=dict[int, VideoJobReject | IcLoraJobReject]
    )


def decide_ic_lora_durations(
    options: list[int],
    *,
    stage1_width: int,
    stage1_height: int,
    fps: int,
    stage_mode: IcLoraStageMode,
    job_filter: LocalJobFilter,
    ic_lora_id: str | None = None,
    stills: int = 0,
    tiling: IcLoraTiling | None = None,
) -> IcLoraDurationOutcome:
    """Decide every duration with one decide call each; the token-ceiling fence rides inside it.

    ``stage_mode``, ``stills`` and ``tiling`` must be what the pipeline runs, or the
    filter prices another job. ``ic_lora_id`` threads to the Mac ingredients term so
    the list-time offer matches the generate-time verdict (offer ≡ verdict); None
    prices on the line, same as generate's custom path.
    """
    kept: list[int] = []
    rejected: dict[int, IcLoraDroppedReason] = {}
    rejects: dict[int, VideoJobReject | IcLoraJobReject] = {}
    for seconds in options:
        frames = compute_num_frames(seconds, max(1, fps))
        decision = decide_ic_lora_job(
            stage1_width,
            stage1_height,
            frames,
            stage_mode=stage_mode,
            job_filter=job_filter,
            ic_lora_id=ic_lora_id,
            stills=stills,
            tiling=tiling,
        )
        if isinstance(decision, IcLoraJobLoad):
            kept.append(seconds)
        else:
            rejected[seconds] = decision.reason
            rejects[seconds] = decision
    return IcLoraDurationOutcome(
        kept=kept,
        rejected=rejected,
        rejects=rejects,
    )


def dropped_duration_message_and_code(
    decision: VideoJobReject | IcLoraJobReject,
    job_filter: LocalJobFilter,
) -> tuple[str, str]:
    """(message, code) for a dropped duration. Fixed copy for the spatial,
    fence, Mac-two-stage, and unsupported verdicts; the shared budget builder
    with the compared figures for stream-over and RAM verdicts."""
    reason = decision.reason
    if reason == "spatial":
        return IC_LORA_V1_ENVELOPE_MESSAGE, IC_LORA_SOURCE_TOO_LARGE
    if reason == "two_stage_over_measured":
        return TWO_STAGE_OVER_MEASURED_MESSAGE, VIDEO_JOB_TOO_LARGE
    if reason == "mac_two_stage_unmeasured":
        return MAC_TWO_STAGE_UNMEASURED_MESSAGE, VIDEO_JOB_TOO_LARGE
    if reason == "refine_decode_floor":
        return REFINE_DECODE_FLOOR_MESSAGE, VIDEO_JOB_TOO_LARGE
    if reason == "mps_sequence_over":
        return MPS_SEQUENCE_OVER_MESSAGE, VIDEO_JOB_TOO_LARGE
    if reason == "unmeasured_cell":
        return UNMEASURED_CELL_MESSAGE, VIDEO_JOB_TOO_LARGE
    if reason == "unsupported":
        return VIDEO_JOB_REJECT_ERRORS["unsupported"]
    assert reason not in _FIXED_COPY_REASONS  # fixed-copy branches took them
    assert isinstance(decision, VideoJobReject)  # only shared reasons reach here
    return reject_message_and_code(decision, job_filter)


def _dropped_duration_list(
    outcome: IcLoraDurationOutcome, *, job_filter: LocalJobFilter
) -> list[DroppedIcLoraDuration]:
    """Catalog-order refusals for every rejected duration."""
    dropped: list[DroppedIcLoraDuration] = []
    for seconds, reason in outcome.rejected.items():
        message, _ = dropped_duration_message_and_code(
            outcome.rejects[seconds], job_filter
        )
        reject = outcome.rejects[seconds]
        risk = (
            risk_for_reject(reject, job_filter)
            if isinstance(reject, VideoJobReject)
            else None
        )
        dropped.append(
            DroppedIcLoraDuration(
                value=seconds,
                reason=reason,
                message=message,
                tier=iclora_drop_risk_tier(reason),
                over_gib=(risk.over_gib if risk is not None else None),
                over_ram_gb=(risk.over_ram_gb if risk is not None else None),
                shortfall_kind=(risk.shortfall_kind if risk is not None else None),
                readout=(
                    risk.readout
                    if risk is not None
                    else untested_readout()
                    if reason in ("two_stage_over_measured", "mac_two_stage_unmeasured", "unmeasured_cell")
                    else None
                ),
            )
        )
    return dropped


def with_budgeted_duration_controls(
    item: IcLoraCatalogItem,
    *,
    job_filter: LocalJobFilter,
) -> IcLoraCatalogItem:
    """Copy ``item`` with duration ``options`` intersected against the job budget.

    Dropped durations ride along in ``dropped_options`` with the refusal the
    list-time gate produced, so pickers can explain instead of silently
    shrinking. Advisory: the gate previews a mid-size canvas without the user
    image, so drops may not hold for small sources — and generate re-decides
    on real pixels, so a dropped duration still 422s-or-passes there. If every
    option drops on a viable machine, options stay full with every refusal
    attached rather than bricking the control on a guess. Unsupported machines
    get empty duration options with every option dropped instead. The preview
    prices the entry's own adapter id, stage layout, stills and tile (offer ≡
    verdict with generate).
    """
    if job_filter.budget.memory_gb is None and job_filter.profile.process_mode != "unsupported":
        # Unknown memory lists unfiltered (generate enforces per job); failing
        # open here avoids bricking the catalog on a probe failure.
        return item

    stage1_w, stage1_h = catalog_preview_stage1(item.default_settings)
    fps = catalog_output_fps(item)
    controls: list[IcLoraControl] = []
    changed = False
    for control in item.controls:
        if control.id != "duration" or control.kind != "int" or control.options is None:
            controls.append(control)
            continue
        listed_options = [o for o in control.options if type(o) is int]
        outcome = decide_ic_lora_durations(
            listed_options,
            stage1_width=stage1_w,
            stage1_height=stage1_h,
            fps=fps,
            stage_mode=item.default_settings.stage_mode,
            job_filter=job_filter,
            ic_lora_id=item.id,
            stills=catalog_preview_stills(item),
            tiling=item.default_settings.tiling,
        )
        if not outcome.rejected:
            controls.append(control)
            continue
        dropped = _dropped_duration_list(outcome, job_filter=job_filter)
        if not outcome.kept:
            if job_filter.profile.process_mode == "unsupported":
                controls.append(
                    control.model_copy(
                        update={"options": [], "dropped_options": dropped}
                    )
                )
                changed = True
                continue
            controls.append(control.model_copy(update={"dropped_options": dropped}))
            changed = True
            continue
        default = control.default
        if not isinstance(default, int) or default not in outcome.kept:
            default = min(outcome.kept)
        controls.append(
            control.model_copy(
                update={
                    "options": outcome.kept,
                    "default": default,
                    "dropped_options": dropped,
                }
            )
        )
        changed = True
    if not changed:
        return item
    return item.model_copy(update={"controls": controls})


def catalog_output_fps(item: IcLoraCatalogItem, loop_fps: int | None = None) -> int:
    """Control-video fps from ``image_to_frames``, or ``DEFAULT_IC_LORA_FPS``.

    ``loop_fps`` overrides the step's rate. An entry without an ``image_to_frames``
    step ignores it.

    Catalog JSON is ``dict[str, JsonValue]``. A None / object / junk ``fps`` must
    not 500 ``GET /api/ic-loras``.
    """
    for step in item.preprocessing:
        if step.utility == "image_to_frames":
            return resolve_loop_fps(step.params.get("fps"), loop_fps)
    return DEFAULT_IC_LORA_FPS
