"""Which 1440p and 4K cells have a measured run behind them.

The budget laws are fitted at and below 1080p. A 1440p or 4K cell that no run has
confirmed prices off an extrapolation, so it is an unmeasured bet: Standard mode
refuses it and Reckless mode allows it with an unmeasured warning. This module is the
only place that says which cells are measured. Adding a row here is how a cell
opens for everyone.
Imports nothing internal by design.
"""

from __future__ import annotations

from typing import Final, Literal

BudgetFeature = Literal["fast", "edit", "iclora"]
BudgetRoute = Literal["mac_measured", "windows_measured", "legacy"]
HighResTier = Literal["1440p", "2160p"]

# Short-side floors that separate the tiers, so the tier follows the resolution name
# at every aspect (a 21:9 1080p canvas is 2560 wide but 1088 high). The canvas short
# edges are 1088 (1080p), 1472 (1440p) and 2176 (2160p). Each floor is the grid step
# below the tier's edge, so only a canvas that is that tier's size or larger counts.
_TIER_MIN_SHORT_SIDE: Final[tuple[tuple[HighResTier, int], ...]] = (
    ("2160p", 2112),
    ("1440p", 1408),
)

# (feature, route, tier) -> the longest job, in windows, that a measured run covers.
# The Windows Fast windowed peak holds flat from the third window and the IC-LoRA tiled peak
# from the fourth, so a row is the longest run measured, not a growth boundary: past it the
# wall-clock and the allocator are untested rather than known to climb. The registry counts
# windows and not their size, so a row cannot say "20 windows at 41 frames but 5 at 129".
MEASURED_HIGH_RES_CELLS: Final[dict[tuple[BudgetFeature, BudgetRoute, HighResTier], int]] = {
    # Mac Fast 4K: capped runs bracket the allocator edge at 2 and 7 windows, and the 7-window
    # one ran to completion, so a longer job is unmeasured. The window count stays at the
    # longest completed run: the Mac peak is flat past the third window, but a 15-window job
    # takes about 6 hours, past any run the heartbeat has covered.
    ("fast", "mac_measured", "2160p"): 7,
    # Windows Fast 1440p: the fence is the 129-frame windows, which a 32 GB card runs and
    # which are measured to five windows. Small cards fall back to 41-frame windows, measured
    # flat to twenty, but a window count cannot tell the two sizes apart, so those jobs past
    # five windows stay unmeasured and refuse in Standard mode. Raising this to twenty would
    # open 129-frame jobs of six to twenty windows that no run covers.
    ("fast", "windows_measured", "1440p"): 5,
    # Windows Fast 4K: held flat from the third window through fifteen (a 50 fps job).
    ("fast", "windows_measured", "2160p"): 15,
    # Windows IC-LoRA 4K: the peak grows through the fourth window and holds to the tenth.
    ("iclora", "windows_measured", "2160p"): 10,
    # Not measured, so absent: Mac 1440p (measured on native runs only, the allocator edge
    # never capped), Mac IC-LoRA (the per-Mpx canvas term is a Windows number), Mac edits
    # above 1080p (past the fitted token range) and any job past the window counts above.
}


# (feature, route) -> (the longest job in windows, the smallest card in GiB) that a measured run
# covers on a canvas below 1440p that runs in windows because one pass cannot fit the card
# (``fast_chunking``). The window rule holds every window to the token size the budget rows were
# fitted at, so the canvas needs no row of its own: the window count and the card are what a run
# confirms. A job in more windows than the row is an unmeasured bet, as past the rows above. A card
# below the run's size, or a route with no row, keeps its single-pass refusals: the small-card
# envelopes were measured on single passes.
MEASURED_WINDOWED_BELOW_1440P_CELLS: Final[dict[tuple[BudgetFeature, BudgetRoute], tuple[int, float]]] = {
    # Windows Fast, 1080p 20 s: one run to completion in 3 windows of 241 frames on the RTX 5090
    # (31.84 GiB). The card floor is a little under that card. Mac has no run.
    ("fast", "windows_measured"): (3, 30.0),
}


# IC-LoRAs that run their own tile and window plan with no run behind it. The rows above
# were fitted on a different composition, so every high cell is unmeasured for these,
# whatever the row says. Opening one means a measured run, then removing its id here.
UNMEASURED_IC_LORA_COMPOSITIONS: Final = frozenset({"refine-details"})


def high_res_tier(width: int, height: int) -> HighResTier | None:
    """The tier of a canvas above 1080p, or None at and below it."""
    short_side = min(width, height)
    for tier, minimum in _TIER_MIN_SHORT_SIDE:
        if short_side >= minimum:
            return tier
    return None


def is_unmeasured_high_res(
    feature: BudgetFeature,
    route: BudgetRoute,
    width: int,
    height: int,
    *,
    windows: int,
    ic_lora_id: str | None = None,
) -> bool:
    """True for a 1440p or 4K job that no measured run covers.

    A job is covered when its (feature, route, tier) has a row and the job runs in no
    more windows than that row's run did. An IC-LoRA in ``UNMEASURED_IC_LORA_COMPOSITIONS``
    is unmeasured in every high cell.
    """
    tier = high_res_tier(width, height)
    if tier is None:
        return False
    if ic_lora_id in UNMEASURED_IC_LORA_COMPOSITIONS:
        return True
    measured_windows = MEASURED_HIGH_RES_CELLS.get((feature, route, tier))
    return measured_windows is None or windows > measured_windows


def window_fallback_measured(feature: BudgetFeature, route: BudgetRoute, total_vram_gib: float | None) -> bool:
    """True when a measured run covers a window fallback below 1440p on a card this size.

    The budget tries windows for a refused job only then. Without a row, below the run's card, or
    with an unknown size, the single-pass verdict stands exactly as before.
    """
    cover = MEASURED_WINDOWED_BELOW_1440P_CELLS.get((feature, route))
    return cover is not None and total_vram_gib is not None and total_vram_gib >= cover[1]


def is_unmeasured_windowed_below_1440p(
    feature: BudgetFeature, route: BudgetRoute, width: int, height: int, *, windows: int
) -> bool:
    """True for a job below 1440p that runs in more windows than a measured run covers.

    One pass is never unmeasured here: it prices on rows fitted at and below 1080p. Only a job in
    more than one window is, and only when its (feature, route) has no row or the job runs in more
    windows than that row's run did. 1440p and 4K have their own rows (``is_unmeasured_high_res``).
    """
    if windows <= 1 or high_res_tier(width, height) is not None:
        return False
    cover = MEASURED_WINDOWED_BELOW_1440P_CELLS.get((feature, route))
    return cover is None or windows > cover[0]
