"""Whole-run wall-time estimates for local Fast video (ETA v1).

Reference table measured on one Windows 11 / RTX 5090 / 63.45 GiB RAM box:
cold whole runs (prompt processing through media encoding) plus the
same-session CPU-vs-DISK control behind the DISK column. Scope is
deliberately narrow:

- exact (width, height, frames) match only — no token interpolation;
- cold column (first-run experience);
- measured hardware class only (``is_measured_hardware_class``);
- I2V/A2V reuse T2V numbers flagged ``t2v_reuse`` (memory-equivalent,
  timing-unmeasured).

Unknown anything -> None. An ETA that says unknown beats one that lies.
``machine_factor`` is the calibration-probe seam: 1.0 is the measured class,
None is an uncalibrated machine. Whether one scalar transfers is exactly what
the clock-cap validation phase is for — this module does not assert it.
Per-stage decomposition is out: capture stage timers do not reconcile with
wall, so the table stays whole-run.
Re-measure rather than trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

EtaBasis = Literal["measured", "scaled"]
EtaFeature = Literal["t2v", "i2v", "a2v"]
EtaLoadMode = Literal["full_models_loading", "streaming_models_loading"]
EtaOffloadMode = Literal["NONE", "CPU", "DISK"]


@dataclass(frozen=True, slots=True)
class WallTimeEstimate:
    seconds: int
    basis: EtaBasis
    t2v_reuse: bool


@dataclass(frozen=True, slots=True)
class _Cell:
    full_cold_s: int | None
    stream_cpu_cold_s: int
    disk_ratio: float
    disk_ratio_measured: bool


# Mean of the measured same-session DISK/CPU ratios. Applied only
# where no direct DISK measurement exists, and labelled "scaled" there.
_DEFAULT_DISK_RATIO = 0.8

# (width, height, frames) at 24 fps. 720p/20s and 1080p/10s never full-load,
# so their full column is None rather than a guess. Keys are fps-agnostic: a
# same-frames job at another rate (5s@25fps is 121f like 5s@24fps) matches the
# same key wherever its (w, h) is listed — no new cells needed there. New
# ratios change (w, h) and need calibration; unlisted stays None.
_TABLE: dict[tuple[int, int, int], _Cell] = {
    (1024, 576, 121): _Cell(121, 158, 0.87, True),
    (1024, 576, 481): _Cell(206, 222, _DEFAULT_DISK_RATIO, False),
    (1280, 704, 481): _Cell(None, 280, 0.81, True),
    (1920, 1088, 121): _Cell(215, 232, _DEFAULT_DISK_RATIO, False),
    (1920, 1088, 241): _Cell(None, 338, _DEFAULT_DISK_RATIO, False),
    # 4K runs in windows and never full-loads. Both cells come from one run each.
    (3840, 2176, 73): _Cell(None, 416, _DEFAULT_DISK_RATIO, False),
    (3840, 2176, 241): _Cell(None, 1537, _DEFAULT_DISK_RATIO, False),
}


def is_measured_hardware_class(device_name: str | None) -> bool:
    """Desktop RTX 5090 class. A substring, deliberately: driver string
    variations must not silently disable the feature, and any 5090-variant
    SKU is close enough to the measured box for a cold whole-run estimate.
    Laptops match the name family but not the clocks, so they are out."""
    if not device_name:
        return False
    lowered = device_name.lower()
    return "5090" in lowered and "laptop" not in lowered


def estimate_fast_wall_time_s(
    *,
    width: int,
    height: int,
    frames: int,
    load_mode: EtaLoadMode,
    offload_mode: EtaOffloadMode,
    feature: EtaFeature,
    machine_factor: float | None,
) -> WallTimeEstimate | None:
    """Cold whole-run seconds for a Fast cell, or None when unknown.

    Incoherent mode combos (full with offload, streamed with NONE) also return
    None: the table has no such measurement and will not invent one.
    """
    if machine_factor is None or machine_factor <= 0:
        return None
    cell = _TABLE.get((width, height, frames))
    if cell is None:
        return None
    if load_mode == "full_models_loading":
        if offload_mode != "NONE" or cell.full_cold_s is None:
            return None
        seconds = cell.full_cold_s
        basis: EtaBasis = "measured"
    elif load_mode == "streaming_models_loading":
        if offload_mode == "CPU":
            seconds = cell.stream_cpu_cold_s
            basis = "measured"
        elif offload_mode == "DISK":
            seconds = round(cell.stream_cpu_cold_s * cell.disk_ratio)
            basis = "measured" if cell.disk_ratio_measured else "scaled"
        else:
            return None
    else:
        assert_never(load_mode)
    return WallTimeEstimate(
        seconds=round(seconds * machine_factor),
        basis=basis,
        t2v_reuse=feature != "t2v",
    )
