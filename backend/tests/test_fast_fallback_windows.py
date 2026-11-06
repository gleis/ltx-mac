"""Fast falls back to windows when one pass cannot fit a card a measured run covers."""

from __future__ import annotations

from itertools import product

import pytest

import runtime_config.video_job_budget as budget
from runtime_config.fast_chunking import is_chunked_canvas, job_window_count, window_candidates
from runtime_config.measured_cells import (
    MEASURED_WINDOWED_BELOW_1440P_CELLS,
    is_unmeasured_windowed_below_1440p,
    window_fallback_measured,
)
from runtime_config.video_job_budget import VideoJobDecision, VideoJobLoad, VideoJobReject, decide_video_job
from tests.test_video_job_budget import _job_filter, _mac_profile, _windows_measured_filter

_1080P = (1920, 1088)
_1080P_21_9 = (2560, 1088)
_3_2 = (1632, 1088)  # 1.78 Mpx: between 720p and 1080p, so no pixel band names it
_720P = (1280, 704)
_540P = (960, 544)
_1440P = (2560, 1472)
_RTX_5090_GIB = 31.84


def _decide(
    size: tuple[int, int], seconds: int, vram_gib: float, fps: int = 24, free_gib: float | None = None
) -> VideoJobDecision:
    return decide_video_job(
        *size, seconds * fps + 1, job_filter=_windows_measured_filter(vram_gib, 64, free_gib=free_gib)
    )


def _single_pass(size: tuple[int, int], seconds: int, vram_gib: float) -> VideoJobDecision:
    """The verdict with the fallback off: what the job got before windows were tried."""
    return budget._decide_fast_pass(  # pyright: ignore[reportPrivateUsage]
        *size, seconds * 24 + 1, windows=1, job_filter=_windows_measured_filter(vram_gib, 64)
    )


def test_1080p_20s_runs_in_three_windows_on_the_measured_card() -> None:
    decision = _decide(_1080P, 20, _RTX_5090_GIB)
    assert isinstance(decision, VideoJobLoad)
    assert decision.chunk_pixel_frames == window_candidates(*_1080P)[0] == 241
    assert job_window_count(20 * 24 + 1, decision.chunk_pixel_frames) == 3


def test_the_rule_names_no_canvas() -> None:
    """A canvas no pixel band named runs in windows once one pass cannot fit the card."""
    one_pass = _single_pass(_3_2, 20, _RTX_5090_GIB)
    assert isinstance(one_pass, VideoJobReject) and one_pass.stream_gib > _RTX_5090_GIB
    decision = _decide(_3_2, 20, _RTX_5090_GIB)
    assert isinstance(decision, VideoJobLoad)
    assert decision.chunk_pixel_frames is not None
    assert not is_chunked_canvas(*_3_2)


@pytest.mark.parametrize("vram_gib", [12, 16, 24, 28])
def test_a_card_below_the_measured_one_keeps_the_single_pass_refusal(vram_gib: float) -> None:
    decision = _decide(_1080P, 20, vram_gib)
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"
    assert decision.chunk_pixel_frames is None


@pytest.mark.parametrize("free_gib", [22.0, 20.0, 16.0])  # one pass of 1080p 10 s asks 22.7 GiB
def test_a_busy_card_keeps_the_single_pass_refusal(free_gib: float) -> None:
    """One pass fits an empty card, so a busy or warm card refuses it. Evict-retry and the
    close-other-apps message stay in charge: windows must not hide a shortfall that frees up."""
    decision = _decide(_1080P, 10, _RTX_5090_GIB, free_gib=free_gib)
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"
    assert decision.chunk_pixel_frames is None


def test_a_job_past_the_measured_window_count_is_unmeasured() -> None:
    # 961 frames (20 s at 48 fps) is 5 windows of 241; the run covered 3.
    decision = _decide(_1080P, 20, _RTX_5090_GIB, fps=48)
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "unmeasured_cell"
    assert decision.chunk_pixel_frames == 241


def test_the_measured_cover_is_a_row_not_the_code() -> None:
    assert MEASURED_WINDOWED_BELOW_1440P_CELLS[("fast", "windows_measured")] == (3, 30.0)
    assert window_fallback_measured("fast", "windows_measured", _RTX_5090_GIB)
    assert not window_fallback_measured("fast", "windows_measured", 24.0)
    assert not window_fallback_measured("fast", "windows_measured", None)
    assert not window_fallback_measured("fast", "mac_measured", 64.0)
    assert not window_fallback_measured("edit", "windows_measured", _RTX_5090_GIB)


def test_one_pass_and_high_res_are_never_unmeasured_here() -> None:
    assert not is_unmeasured_windowed_below_1440p("fast", "mac_measured", *_1080P, windows=1)
    assert is_unmeasured_windowed_below_1440p("fast", "mac_measured", *_1080P, windows=2)
    assert not is_unmeasured_windowed_below_1440p("fast", "windows_measured", *_720P, windows=3)
    assert is_unmeasured_windowed_below_1440p("fast", "windows_measured", *_720P, windows=4)
    # 1440p and 4K have their own rows (is_unmeasured_high_res).
    assert is_chunked_canvas(*_1440P)
    assert not is_unmeasured_windowed_below_1440p("fast", "windows_measured", *_1440P, windows=9)


@pytest.mark.parametrize("pool_gib", [16, 32, 48, 96])
def test_mac_keeps_every_single_pass_refusal(pool_gib: float) -> None:
    """Mac has no row, so no window is tried below 1440p: its refusals stay single-pass."""
    jf = _job_filter(memory_gb=pool_gib * 2, process_mode="full_models_loading", **_mac_profile(pool_gib=pool_gib))
    for size, seconds in product((_540P, _720P, _1080P, _1080P_21_9), (10, 20)):
        decision = decide_video_job(*size, seconds * 24 + 1, job_filter=jf)
        assert decision.chunk_pixel_frames is None, (size, seconds)


def test_no_job_that_loads_or_is_refused_for_another_reason_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback only turns a refusal no empty card could pass into a windowed verdict."""
    sizes = [_540P, _720P, _3_2, _1080P, _1080P_21_9, _1440P]
    cards = [(12, None), (24, None), (_RTX_5090_GIB, None), (48, None), (_RTX_5090_GIB, 20.0), (48, 30.0)]
    cases = list(product(sizes, (5, 8, 10, 20), (24, 48), cards))

    def run() -> dict[tuple, VideoJobDecision]:
        return {c: _decide(c[0], c[1], c[3][0], fps=c[2], free_gib=c[3][1]) for c in cases}

    after = run()
    monkeypatch.setattr(budget, "window_fallback_measured", lambda *_a, **_k: False)
    before = run()

    changed = [c for c in cases if before[c] != after[c]]
    assert any(isinstance(after[c], VideoJobLoad) for c in changed), "a refused job should open"
    for c in changed:
        was, now = before[c], after[c]
        total = c[3][0]
        assert isinstance(was, VideoJobReject), c
        assert was.reason == "stream_over", c
        assert was.stream_gib > total, c  # no empty card of this size runs it in one pass
        assert total >= 30.0, c
        assert not is_chunked_canvas(*c[0]), c
        if isinstance(now, VideoJobLoad):
            assert now.chunk_pixel_frames is not None, c
        else:
            assert now.reason == "unmeasured_cell", c  # past the windows the run covered
