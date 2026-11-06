"""Shared job-reject reason vocabulary.

One definition imported by the budget layer (decision types, HTTP mapping)
and the API layer (specs reason field), so a new reason flows through both
without a second Literal drifting. Imports nothing internal by design.
"""

from __future__ import annotations

from typing import Literal

# ``unmeasured_cell``: a 1440p or 4K cell no run has confirmed (measured_cells.py).
# Standard refuses it with fixed copy; Reckless allows it with an unmeasured warning.
VideoJobRejectReason = Literal[
    "stream_over", "unsupported", "ram_shortfall", "mps_sequence_over", "unmeasured_cell"
]

# Hardware-budget mode: which measurement tiers block. "reckless" turns an
# over-budget verdict into allow-with-warning and annotates unmeasured loads.
# An unmeasured note does not cover a later memory miss. Hard refusals
# (unsupported, fences, window math) block in every mode. Stored in
# AppSettings, threaded onto LocalJobFilter, echoed on specs responses.
BudgetMode = Literal["standard", "reckless"]

# Warning tiers for allow-with-warning UX. A property of the (job, machine)
# pair, never of the mode: the mode only decides which tiers block.
LoadRiskTier = Literal["tight", "unmeasured"]

# Tiers a create-time annotation may carry. Tight fits never block anywhere,
# so they are never annotated.
JobRiskTier = Literal["over", "unmeasured"]

# Which figure bound a CUDA over-budget verdict. One spelling shared by the
# dropped cells, the risk annotation, and the budget helpers.
ShortfallKind = Literal["total", "free"]

# List-time IC-LoRA duration drops: the shared budget reasons plus the four
# IC-LoRA-only verdicts (spatial canvas fence, unpriced two-stage lengths,
# Mac two-stage refused until measured, long Refine past the decode floor).
# Kept separate from VideoJobRejectReason because the shared vocabulary feeds
# the Fast/edit reject table, where these four can never occur. The budget
# reasons still pass through by name, so extending the shared literal fails
# typecheck at the IC-LoRA passthrough until this literal follows.
IcLoraDroppedReason = Literal[
    "stream_over",
    "unsupported",
    "ram_shortfall",
    "mps_sequence_over",
    "unmeasured_cell",
    "spatial",
    "two_stage_over_measured",
    "mac_two_stage_unmeasured",
    "refine_decode_floor",
]

# Execute-time budget family: HTTP/capability codes meaning "the job stopped
# fitting between create and execute". The queue runner maps these to
# JOB_TOO_LARGE; the frontend passes them through verbatim. Literal strings,
# not imports: this module imports nothing internal by design (the budget
# constants live downstream). Pinned to shared/budget-miss-codes.json by
# test_budget_miss_codes_match_shared_contract — extend both together.
BUDGET_MISS_CODES: frozenset[str] = frozenset(
    {
        "IC_LORA_SOURCE_TOO_LARGE",
        "LOCAL_GENERATION_RAM_SHORTFALL",
        "LOCAL_GENERATION_UNSUPPORTED",
        "VIDEO_JOB_TOO_LARGE",
    }
)
