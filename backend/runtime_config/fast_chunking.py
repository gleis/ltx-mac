"""Temporal windows for large Fast video jobs (1440p and 4K, and any canvas one pass cannot fit).

The measured Fast budget is calibrated up to a window of about 63k tokens. A 1440p clip of
10 s is 114k tokens and a 4K clip of 10 s is 253k, so a large canvas runs in overlapping
windows (ltx-pipelines ``ChunkConfig``) and the job budget prices one window, not the clip.
This module is the one rule for the window. The job budget (``decide_video_job``) picks the
largest window that loads and hands its length to the pipeline.

It imports nothing from the budget, so the budget can import it.
"""

from __future__ import annotations

import math
from typing import Final

# Canvas pixels above which a clip always runs in windows. A smaller canvas stays on the single-pass
# rows, and runs in windows only when one pass cannot fit the card (``decide_video_job``). The
# smallest 1440p canvas is 3.77M pixels.
CHUNKED_CANVAS_MIN_PIXELS: Final = 3_200_000

# The window may not carry more than this many tokens: it is the largest size the budget
# rows were calibrated on.
MAX_WINDOW_TOKENS: Final = 63_240

# ltx-pipelines rejects a carry below three latent frames. The window must also be more than
# twice the carry, and both must sit on the 8k + 1 grid.
CARRY_FRAMES: Final = 17
MIN_WINDOW_FRAMES: Final = 41
_FRAMES_PER_LATENT: Final = 8

# Windows from this count on hold their peak: the allocator keeps what the carry and the
# earlier windows left behind. A shorter run peaks lower.
PEAK_PLATEAU_WINDOWS: Final = 3


def is_chunked_canvas(width: int, height: int) -> bool:
    return width * height >= CHUNKED_CANVAS_MIN_PIXELS


def _tokens_per_latent_frame(width: int, height: int) -> float:
    return (height / 32.0) * (width / 32.0)


def window_candidates(width: int, height: int) -> list[int]:
    """Window lengths in pixel frames, longest first, one latent frame apart.

    The longest is the largest that stays inside ``MAX_WINDOW_TOKENS``. A machine that cannot
    load it takes the next one down, to ``MIN_WINDOW_FRAMES``.
    """
    latent_frames = int(MAX_WINDOW_TOKENS // _tokens_per_latent_frame(width, height))
    longest = max(MIN_WINDOW_FRAMES, (latent_frames - 1) * _FRAMES_PER_LATENT + 1)
    return list(range(longest, MIN_WINDOW_FRAMES - 1, -_FRAMES_PER_LATENT))


def window_count(frames: int, window_frames: int, carry_frames: int = CARRY_FRAMES) -> int:
    """How many windows a clip of ``frames`` runs in. Each later window adds ``window - carry``."""
    if frames <= window_frames:
        return 1
    return 1 + math.ceil((frames - window_frames) / (window_frames - carry_frames))


def job_window_count(
    frames: int, chunk_pixel_frames: int | None, carry_frames: int = CARRY_FRAMES
) -> int:
    """Windows of a decided job: one pass when no window was chosen.

    ``carry_frames`` is the carry the pipeline runs with. Fast's is ``CARRY_FRAMES``. An
    IC-LoRA passes its own (``ic_lora_carry_frames``), or its jobs are miscounted.
    """
    if chunk_pixel_frames is None:
        return 1
    return window_count(frames, chunk_pixel_frames, carry_frames)
