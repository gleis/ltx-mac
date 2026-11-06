"""Window choice of ``decide_video_job`` for 1440p and 4K (Windows 5090-class budget).

Each test names the bug it guards. The VRAM values sit between the measured-law asks:
a 4K window of 49 frames asks 21.0 GiB and a window of 41 frames asks 18.7 GiB. Two
windows add 6.5e-5 GiB per window token and 3 or more add 1.05e-4.
"""

from __future__ import annotations

from frame_math import compute_num_frames
from runtime_config.fast_chunking import window_count
from runtime_config.measured_cells import MEASURED_HIGH_RES_CELLS
from runtime_config.video_job_budget import (
    VideoJobLoad,
    VideoJobReject,
    advertised_fast_durations,
    decide_video_job,
    fast_job_stream_gib,
    video_tokens,
)
from tests.test_video_job_budget import _windows_measured_filter

_4K = (3840, 2176)
_1440P = (2560, 1472)
_DURATIONS = [2, 3, 4, 5, 6, 8, 10, 20]


def _decide(size: tuple[int, int], frames: int, vram_gib: float):
    return decide_video_job(
        *size, frames, job_filter=_windows_measured_filter(vram_gib, 64)
    )


def test_clip_that_fits_one_window_is_not_chunked() -> None:
    decision = _decide(_4K, 49, 32)
    assert isinstance(decision, VideoJobLoad)
    assert decision.chunk_pixel_frames is None


def test_clip_over_one_window_takes_the_longest_window_not_the_shortest() -> None:
    decision = _decide(_4K, 73, 32)
    assert isinstance(decision, VideoJobLoad)
    assert decision.chunk_pixel_frames == 49


def test_card_too_small_for_the_long_window_falls_back_to_a_shorter_one() -> None:
    # 65 frames is 2 windows either way. 23 GiB rejects the 49-frame windows (24.7 GiB)
    # and loads the 41-frame windows (21.9 GiB).
    decision = _decide(_4K, 65, 23)
    assert isinstance(decision, VideoJobLoad)
    assert decision.chunk_pixel_frames == 41


def test_three_windows_pay_a_larger_extra_than_two_windows() -> None:
    tokens = video_tokens(_4K[1], _4K[0], 49)
    one = fast_job_stream_gib(tokens, windows=1, darwin=False)
    two = fast_job_stream_gib(tokens, windows=2, darwin=False)
    three = fast_job_stream_gib(tokens, windows=3, darwin=False)
    assert one < two < three


def test_two_window_job_with_full_last_window_asks_the_second_window_step() -> None:
    # 1440p at 10 s and 24 fps is 129 frames then 112 more: window 2 peaked 25.6 GiB.
    # A +0 two-window price asked 22.5 and let a 24 GiB card admit a job that pages.
    tokens = video_tokens(_1440P[1], _1440P[0], 129)
    assert window_count(241, 129) == 2
    assert fast_job_stream_gib(tokens, windows=2, darwin=False) >= 25.6


def test_plateau_extra_scales_with_window_tokens_not_a_fixed_constant() -> None:
    # A fixed 5.7 asked 16.9 of 41-frame windows that peaked 13.04, refusing a 16 GB card's
    # job, and asked 28.2 of 1440p's 129-frame windows that peaked 28.31.
    small = fast_job_stream_gib(video_tokens(_1440P[1], _1440P[0], 41), windows=20, darwin=False)
    assert 13.04 <= small < 16.88
    large = fast_job_stream_gib(video_tokens(_1440P[1], _1440P[0], 129), windows=5, darwin=False)
    assert large >= 28.31


def test_extra_does_not_grow_with_window_count_past_three() -> None:
    # The peak is flat from window 3 to 20; a per-window term would price long jobs out.
    tokens = video_tokens(_4K[1], _4K[0], 49)
    three = fast_job_stream_gib(tokens, windows=3, darwin=False)
    assert fast_job_stream_gib(tokens, windows=20, darwin=False) == three


def test_fast_4k_row_loads_at_15_windows_and_refuses_at_16() -> None:
    # 497 frames is 15 windows of 49; 529 is 16. The row moved from 7 to 15 on the Fast key only.
    assert MEASURED_HIGH_RES_CELLS[("fast", "windows_measured", "2160p")] == 15
    at_row = _decide(_4K, 497, 64)
    assert isinstance(at_row, VideoJobLoad)
    assert window_count(497, 49) == 15
    past_row = _decide(_4K, 529, 64)
    assert isinstance(past_row, VideoJobReject)
    assert past_row.reason == "unmeasured_cell"


def test_fast_1440p_row_is_the_129_frame_window_count_not_the_41_frame_one() -> None:
    # 481 frames is 5 windows of 129 (measured); 689 is 6. Opening the row to the 41-frame
    # window's 20 would admit 129-frame jobs of 6 to 20 windows that no run covers.
    at_row = _decide(_1440P, 481, 64)
    assert isinstance(at_row, VideoJobLoad)
    assert at_row.chunk_pixel_frames == 129
    past_row = _decide(_1440P, 689, 64)
    assert isinstance(past_row, VideoJobReject)
    assert past_row.reason == "unmeasured_cell"


def test_reject_names_the_ask_of_the_shortest_window() -> None:
    frames = 241
    decision = _decide(_4K, frames, 14)
    assert isinstance(decision, VideoJobReject)
    shortest = 41
    expected = fast_job_stream_gib(
        video_tokens(_4K[1], _4K[0], shortest),
        windows=window_count(frames, shortest),
        darwin=False,
    )
    assert decision.stream_gib == expected


def test_offered_durations_match_the_create_verdict_at_1440p_and_4k() -> None:
    for size, vram in ((_1440P, 16), (_4K, 24), (_4K, 28)):
        job_filter = _windows_measured_filter(vram, 64)
        offered = advertised_fast_durations(*size, 24, _DURATIONS, job_filter=job_filter)
        loads = [
            seconds
            for seconds in _DURATIONS
            if isinstance(
                decide_video_job(
                    *size, compute_num_frames(seconds, 24), job_filter=job_filter
                ),
                VideoJobLoad,
            )
        ]
        assert offered == loads
