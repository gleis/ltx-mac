"""ETA v1 table pins: exact cells, cold column, honest Nones."""

from __future__ import annotations

import pytest

from frame_math import compute_num_frames
from runtime_config.eta import (
    WallTimeEstimate,
    estimate_fast_wall_time_s,
    is_measured_hardware_class,
)

# (width, height, seconds, full_cold, stream_cpu_cold)
_CELLS = [
    (1024, 576, 5, 121, 158),
    (1024, 576, 20, 206, 222),
    (1280, 704, 20, None, 280),
    (1920, 1088, 5, 215, 232),
    (1920, 1088, 10, None, 338),
]


@pytest.mark.parametrize(("width", "height", "seconds", "full", "cpu"), _CELLS)
def test_table_pins_cold_seconds(
    width: int, height: int, seconds: int, full: int | None, cpu: int
) -> None:
    frames = compute_num_frames(seconds, 24)
    stream = estimate_fast_wall_time_s(
        width=width,
        height=height,
        frames=frames,
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature="t2v",
        machine_factor=1.0,
    )
    assert stream == WallTimeEstimate(seconds=cpu, basis="measured", t2v_reuse=False)
    full_est = estimate_fast_wall_time_s(
        width=width,
        height=height,
        frames=frames,
        load_mode="full_models_loading",
        offload_mode="NONE",
        feature="t2v",
        machine_factor=1.0,
    )
    if full is None:
        assert full_est is None
    else:
        assert full_est == WallTimeEstimate(
            seconds=full, basis="measured", t2v_reuse=False
        )


@pytest.mark.parametrize(
    ("width", "height", "seconds", "expected", "basis"),
    [
        # Same-session measured ratios: 0.87 (540p/5s), 0.81 (720p/20s).
        (1024, 576, 5, round(158 * 0.87), "measured"),
        (1280, 704, 20, round(280 * 0.81), "measured"),
        # Default 0.8 ratio elsewhere, labelled scaled.
        (1024, 576, 20, round(222 * 0.8), "scaled"),
        (1920, 1088, 5, round(232 * 0.8), "scaled"),
        (1920, 1088, 10, round(338 * 0.8), "scaled"),
    ],
)
def test_disk_applies_ratio_with_basis(
    width: int, height: int, seconds: int, expected: int, basis: str
) -> None:
    est = estimate_fast_wall_time_s(
        width=width,
        height=height,
        frames=compute_num_frames(seconds, 24),
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        feature="t2v",
        machine_factor=1.0,
    )
    assert est is not None
    assert est.seconds == expected
    assert est.basis == basis


def test_unlisted_cell_returns_none() -> None:
    """720p/10s is a real grid cell with no measurement: unknown, not interpolated."""
    est = estimate_fast_wall_time_s(
        width=1280,
        height=704,
        frames=compute_num_frames(10, 24),
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature="t2v",
        machine_factor=1.0,
    )
    assert est is None


def test_incoherent_mode_combos_return_none() -> None:
    frames = compute_num_frames(5, 24)
    # Full-resident never offloads; streamed is never NONE.
    assert (
        estimate_fast_wall_time_s(
            width=1024,
            height=576,
            frames=frames,
            load_mode="full_models_loading",
            offload_mode="CPU",
            feature="t2v",
            machine_factor=1.0,
        )
        is None
    )
    assert (
        estimate_fast_wall_time_s(
            width=1024,
            height=576,
            frames=frames,
            load_mode="streaming_models_loading",
            offload_mode="NONE",
            feature="t2v",
            machine_factor=1.0,
        )
        is None
    )


def test_unknown_machine_returns_none() -> None:
    est = estimate_fast_wall_time_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature="t2v",
        machine_factor=None,
    )
    assert est is None


def test_machine_factor_scales_table() -> None:
    """The calibration-probe seam: a 2×-slower machine doubles the table."""
    est = estimate_fast_wall_time_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature="t2v",
        machine_factor=2.0,
    )
    assert est == WallTimeEstimate(seconds=316, basis="measured", t2v_reuse=False)


@pytest.mark.parametrize("factor", [0.0, -1.0])
def test_non_positive_factor_returns_none(factor: float) -> None:
    est = estimate_fast_wall_time_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature="t2v",
        machine_factor=factor,
    )
    assert est is None


@pytest.mark.parametrize(
    ("feature", "reuse"),
    [("t2v", False), ("i2v", True), ("a2v", True)],
)
def test_i2v_a2v_reuse_t2v_numbers_flagged(feature: str, reuse: bool) -> None:
    est = estimate_fast_wall_time_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature=feature,  # type: ignore[arg-type]
        machine_factor=1.0,
    )
    assert est == WallTimeEstimate(seconds=158, basis="measured", t2v_reuse=reuse)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NVIDIA GeForce RTX 5090", True),
        ("nvidia geforce rtx 5090", True),
        ("NVIDIA GeForce RTX 5090 Laptop GPU", False),
        ("NVIDIA GeForce RTX 4090", False),
        (None, False),
        ("", False),
    ],
)
def test_measured_hardware_class(name: str | None, expected: bool) -> None:
    assert is_measured_hardware_class(name) is expected
