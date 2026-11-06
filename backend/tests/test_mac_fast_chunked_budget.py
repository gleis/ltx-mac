"""Mac Fast windowed price at 1440p and 4K (``decide_video_job`` on the Darwin route).

Each test names the bug it guards. The Mac law prices the smallest allocator cap that still
passes, so the reference points are capped runs: at 4K a cap of 22.7 GiB failed and 24.0
passed for two windows, and 28.4 failed and 30.0 passed for seven.
"""

from __future__ import annotations

import pytest

from runtime_config.measured_cells import MEASURED_HIGH_RES_CELLS
from runtime_config.video_job_budget import (
    VideoJobLoad,
    VideoJobReject,
    _mac_fast_stream_required_gib,
    _windows_fast_reserved_required_gib,
    decide_video_job,
    fast_job_stream_gib,
    video_tokens,
)
from tests.test_video_job_budget import _job_filter, _mac_profile

_4K = (3840, 2176)
_1440P = (2560, 1472)
_4K_WINDOW_TOKENS = video_tokens(_4K[1], _4K[0], 49)
_1440P_WINDOW_TOKENS = video_tokens(_1440P[1], _1440P[0], 129)


def _decide(size: tuple[int, int], frames: int, pool_gib: float):
    job_filter = _job_filter(
        process_mode="streaming_models_loading", memory_gb=None, **_mac_profile(pool_gib=pool_gib)
    )
    return decide_video_job(*size, frames, job_filter=job_filter)


def _mac_ask(tokens: float, windows: int) -> float:
    return fast_job_stream_gib(tokens, windows=windows, darwin=True)


def test_mac_plateau_ask_covers_the_capped_4k_edge() -> None:
    # A cap of 28.4 GiB failed a 7-window 4K job and 30.0 passed. The fixed 5.7 asked 28.39,
    # and the Windows coefficient put on the Mac asks 28.7: both admit a pool that fails.
    assert _mac_ask(_4K_WINDOW_TOKENS, 7) >= 30.0
    assert _mac_ask(_4K_WINDOW_TOKENS, 7) == _mac_ask(_4K_WINDOW_TOKENS, 3)
    assert isinstance(_decide(_4K, 241, 30.2), VideoJobLoad)


def test_mac_two_window_ask_pays_the_second_window_step() -> None:
    # A cap of 22.7 GiB failed the 73-frame 4K job (2 windows) and 24.0 passed. A +0
    # two-window price asked 22.69 there, and 24.51 for a 1440p 241-frame job whose second
    # window peaked about 3.7 GiB over its first.
    assert _mac_ask(_4K_WINDOW_TOKENS, 2) >= 24.0
    assert _mac_ask(_1440P_WINDOW_TOKENS, 2) >= _mac_ask(_1440P_WINDOW_TOKENS, 1) + 3.72
    refused = _decide(_4K, 73, 22.7)
    assert isinstance(refused, VideoJobReject)
    assert refused.reason == "stream_over"


def test_mac_windowed_extra_scales_with_window_tokens() -> None:
    # A fixed Mac extra asks the same for a 41-frame window as for a 129-frame one, and
    # under-asks the larger 1440p window against the 4K one.
    small = video_tokens(_1440P[1], _1440P[0], 41)

    def extra(tokens: float) -> float:
        return _mac_ask(tokens, 3) - _mac_ask(tokens, 1)

    assert extra(small) < extra(_4K_WINDOW_TOKENS) < extra(_1440P_WINDOW_TOKENS)
    assert extra(_1440P_WINDOW_TOKENS) / _1440P_WINDOW_TOKENS == pytest.approx(
        extra(_4K_WINDOW_TOKENS) / _4K_WINDOW_TOKENS
    )


def test_mac_single_window_ask_is_the_base_law() -> None:
    # The extra starts at the second window. Applied at one window it re-prices every
    # single-pass cell on the Mac.
    for tokens in (_4K_WINDOW_TOKENS, _1440P_WINDOW_TOKENS):
        assert _mac_ask(tokens, 1) == _mac_fast_stream_required_gib(tokens)


def test_windows_and_mac_windowed_coefficients_stay_separate() -> None:
    # One shared constant or branch would move both routes when either is refit.
    tokens = _4K_WINDOW_TOKENS
    for windows, mac_coefficient, windows_coefficient in ((2, 7.5e-5, 6.5e-5), (3, 1.30e-4, 1.05e-4)):
        mac_extra = _mac_ask(tokens, windows) - _mac_fast_stream_required_gib(tokens)
        assert mac_extra == pytest.approx(mac_coefficient * tokens)
        windows_extra = fast_job_stream_gib(
            tokens, windows=windows, darwin=False
        ) - _windows_fast_reserved_required_gib(tokens)
        assert windows_extra == pytest.approx(windows_coefficient * tokens)


def test_mac_fast_4k_row_loads_at_7_windows_and_refuses_at_8() -> None:
    # 241 frames is 7 windows of 49, 249 is 8. The row is the longest completed run: a row
    # on the wrong tier, feature or route, or off by one, moves this boundary.
    assert MEASURED_HIGH_RES_CELLS[("fast", "mac_measured", "2160p")] == 7
    at_row = _decide(_4K, 241, 107.52)
    assert isinstance(at_row, VideoJobLoad)
    assert at_row.chunk_pixel_frames == 49
    past_row = _decide(_4K, 249, 107.52)
    assert isinstance(past_row, VideoJobReject)
    assert past_row.reason == "unmeasured_cell"
    assert ("iclora", "mac_measured", "2160p") not in MEASURED_HIGH_RES_CELLS


@pytest.mark.parametrize("frames", [121, 241, 481])
def test_mac_fast_1440p_stays_unmeasured_at_every_window_count(frames: int) -> None:
    # 1 window (121 frames), 2 windows (241) and 5 windows (481) of 129 frames were run
    # native only; the allocator edge was never capped. A Mac 1440p row, or the 4K row
    # keyed on the wrong tier, would open these in Standard mode.
    refused = _decide(_1440P, frames, 107.52)
    assert isinstance(refused, VideoJobReject)
    assert refused.reason == "unmeasured_cell"
