"""Tile plan and Windows budget of the 4K IC-LoRA cell (any catalog LoRA, tiled)."""

from __future__ import annotations

from runtime_config.ic_lora_high_res import high_res_tiling
from runtime_config.ic_lora_job_budget import IcLoraJobLoad, IcLoraJobReject, decide_ic_lora_job
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.ic_lora_tiling import IcLoraTiling
from runtime_config.measured_cells import MEASURED_HIGH_RES_CELLS
from runtime_config.video_job_budget import VideoJobReject
from tests.test_video_job_budget import _windows_measured_filter

_4K = (3840, 2176)
_FRAMES_10S = 233


def _decide_4k(vram_gib: float, tiling: IcLoraTiling):
    return decide_ic_lora_job(
        *_4K,
        _FRAMES_10S,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(vram_gib, 64),
        tiling=tiling,
    )


def test_lora_without_a_tile_size_runs_the_local_cap_not_a_small_tile() -> None:
    # A 1024x576 tile needed 6x6 tiles at 4K and ran 210 s per step on the RTX 5090.
    tiling = high_res_tiling(None)
    assert (tiling.long_side, tiling.short_side) == (1920, 1088)


def test_lora_with_its_own_tile_size_keeps_it_in_the_high_cell() -> None:
    own = IcLoraTiling(long_side=960, short_side=544)
    tiling = high_res_tiling(own)
    assert (tiling.long_side, tiling.short_side) == (960, 544)


def test_4k_ten_second_job_loads_on_a_32gb_card() -> None:
    # The 25% margin on the canvas term priced this job at 32.1 GiB and refused it.
    decision = _decide_4k(31.84, high_res_tiling(None))
    assert isinstance(decision, IcLoraJobLoad)


def test_4k_ten_second_job_is_refused_on_a_card_below_its_ask() -> None:
    decision = _decide_4k(28, high_res_tiling(None))
    # The decode of the job's last window no longer fits, so the decode check refuses it before
    # the stream curve does.
    assert isinstance(decision, (VideoJobReject, IcLoraJobReject))


def _decide_4k_frames(frames: int):
    return decide_ic_lora_job(
        *_4K,
        frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(64, 64),
        tiling=high_res_tiling(None),
    )


def test_iclora_4k_row_loads_at_10_windows_and_refuses_at_11() -> None:
    # 257 frames is 10 windows of 41; 281 is 11. Only the IC-LoRA key moves 9 to 10: the
    # Fast 4K row is 15 and the IC-LoRA one must not follow it.
    assert MEASURED_HIGH_RES_CELLS[("iclora", "windows_measured", "2160p")] == 10
    assert isinstance(_decide_4k_frames(257), IcLoraJobLoad)
    past_row = _decide_4k_frames(281)
    assert isinstance(past_row, VideoJobReject)
    assert past_row.reason == "unmeasured_cell"


_4K_REFINE = (3840, 2112)
_REFINE_TILING = high_res_tiling(IcLoraTiling(long_side=1024, short_side=576, window_frames=73))


def _decide_refine_4k(vram_gib: float, frames: int):
    return decide_ic_lora_job(
        *_4K_REFINE,
        frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(vram_gib, 64),
        tiling=_REFINE_TILING,
    )


def test_refine_4k_137_frames_runs_49_frame_windows_on_a_32gb_card() -> None:
    # The decode budget read 27.8, 22.7 and 20.9 GiB on windows 1 to 3 of this job at 73 frames.
    # At 20.9 GiB a 73-frame decode does not fit, so a third full window would fail after denoise.
    decision = _decide_refine_4k(31.84, 137)
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.chunk_pixel_frames == 49


def test_refine_4k_one_window_clip_keeps_its_single_pass_on_a_32gb_card() -> None:
    # Window 1 reads 27.8 GiB and a 73-frame decode needs 22.3, so the clip runs as it did.
    decision = _decide_refine_4k(31.84, 73)
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.chunk_pixel_frames is None


def test_refine_4k_is_refused_before_denoise_when_no_window_decodes() -> None:
    decision = _decide_refine_4k(24, 137)
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "refine_decode_floor"


def test_refine_4k_keeps_its_catalog_window_on_a_card_with_room() -> None:
    decision = _decide_refine_4k(48, 137)
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.chunk_pixel_frames == 73
