"""Accelerator backend detection.

ROCm PyTorch reports itself through the same `torch.cuda` interface as NVIDIA
CUDA builds (`torch.cuda.is_available()`, `device.type == "cuda"`, etc.), so
code that checks `device.type == "cuda"` to mean "this is an NVIDIA GPU" is
wrong under ROCm. `torch.version.hip` is the actual discriminator: it is set
on ROCm builds and `None` on CUDA builds.

Originally contributed by boxwrench in
https://github.com/Lightricks/LTX-Desktop/pull/160
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

AcceleratorBackend = Literal["rocm", "cuda", "mps", "cpu"]

# First compute capability with native fp8 (E4M3/E5M2): Ada Lovelace.
# Ampere (8.x) and Turing (7.5) have no hardware fp8.
_FP8_MIN_COMPUTE_CAPABILITY = (8, 9)


def accelerator_backend() -> AcceleratorBackend:
    if getattr(torch.version, "hip", None):
        return "rocm"

    if torch.cuda.is_available():
        return "cuda"

    if mps_available():
        return "mps"

    return "cpu"


def mps_available() -> bool:
    """Whether the MPS backend exists and is usable. False everywhere off-Mac."""
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and bool(mps.is_available())


def accelerator_supports_fp8(device_index: int = 0) -> bool:
    """Whether a CUDA device can run the fp8 transformer path.

    True only on NVIDIA CUDA with compute capability >= 8.9. ROCm, MPS, CPU,
    pre-Ada NVIDIA, and any probe failure all fail closed to False: without
    fp8 the full bf16 transformer cannot stay resident at the full-loading
    floor, so callers must take the streaming / unquantized path. Device 0 by
    default, matching the total- and free-VRAM probes.
    """
    if accelerator_backend() != "cuda":
        return False
    try:
        return torch.cuda.get_device_capability(device_index) >= _FP8_MIN_COMPUTE_CAPABILITY
    except Exception:
        logger.warning("fp8 capability probe failed; treating device as non-fp8", exc_info=True)
        return False
