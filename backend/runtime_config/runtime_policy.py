"""Runtime policy decisions for local generation mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    # Runtime import would cycle: api_types initializes through this module
    # while services/__init__ needs api_types (see runtime_config.py).
    from services.storage_info.storage_info import DriveType

LocalGenerationMode = Literal[
    "full_models_loading",
    "streaming_models_loading",
    "mac_mlx_q4",
    "unsupported",
]

# Documented Darwin minimum (README, compat matrix). Keep product copy here.
DARWIN_ADVERTISED_FLOOR_GB = 32

# Documented macOS minimum for local generation (README). The bundled prebuilt
# mpsgraph_zc extension is built for macOS 26 and cannot load on older systems.
# On macOS 15, mps-sdpa falls back to a backend that leaks Metal memory and
# runs out of memory on longer clips (docs/mps-attention-memory-leak.md).
# LTX-Desktop#171 reports a failure at the first attention call on macOS 14.8.2.
# The cause is NOT confirmed. One hypothesis is that the MPSGraph attention API
# needs macOS 15. Do not treat it as verified evidence for this floor.
DARWIN_MIN_MACOS_MAJOR = 26

# Unpublished API-only cutoff (integer GiB total RAM). Do not use in user-facing
# copy — ``DARWIN_ADVERTISED_FLOOR_GB`` is the documented minimum.
DARWIN_STREAMING_FLOOR_GB = 32

# CUDA discrete VRAM floors, compared against exact GiB (bytes-derived via
# ``MemoryBudget.total_vram_gib`` at the production path; plain integers compare
# identically against integer floors). Below the streaming floor local gen is not
# viable. At/above the full floor, fp8-capable cards can hold the halved transformer
# resident. The streaming floor is 9: an early-out, not a policy — derived, not
# picked. The per-cell model is the real gate (it refuses unfitting cells against
# free VRAM per request), so the floor only skips grid evaluation on hardware that
# cannot run the smallest cell: 8.00 GiB (540p/5s, Phase 2 bracket (7.9, 8.0]) +
# ~1.00 GiB typical desktop occupancy (Phase 0). Cards report below nominal (16 GB
# cards report 15.99 — see ``test_gpu_raw_capacity``), so 10 GB-class cards at
# ~9.99 stream and the model offers them ~8 cells, while 8 GB-class cards at ~7.99
# stay excluded. Per-cell budgets (not this floor) decide what each card offers.
# The 1.00 GiB desktop figure is an estimate (no Phase 0 survey was captured) —
# do not tighten this floor toward 8.00 on its strength. fp8 cards only: without
# fp8 the bf16 path is unmeasured, so non-fp8 CUDA keeps the 15 floor below.
CUDA_VRAM_FLOOR_GB = 9
# CUDA_VRAM_FLOOR_GB is Windows-fp8-measured only (5090 laws; the bf16
# path is unmeasured and the legacy curve under-asks 9-14 GB cards). Linux
# and non-fp8 Windows ride the legacy curve below 15 and keep the old floor.
LINUX_VRAM_FLOOR_GB = 15
CUDA_FULL_VRAM_GB = 31

# CUDA OffloadMode.CPU pre-pins every transformer block in host RAM. ltx-pipelines
# documents that path as ~36 GB RAM + ~5 GB VRAM. On Linux those buffers are
# page-locked (unswappable), so a 32 GB machine — the advertised local-gen floor —
# dies in the kernel OOM killer with VRAM still empty (LTX-Desktop#163). Below this
# total-RAM floor, stream from disk instead (OffloadMode.DISK, ~5 GB host).
CUDA_CPU_OFFLOAD_RAM_FLOOR_GB = 40


def decide_local_generation_mode(
    system: str,
    cuda_available: bool,
    vram_gb: float | None,
    mps_available: bool = False,
    ram_gb: int | None = None,
    fp8_capable: bool = True,
    macos_major: int | None = None,
) -> LocalGenerationMode:
    """Pick the local-generation mode for this runtime.

    - "unsupported": local generation is not viable; caller must route to the API.
    - "mac_mlx_q4": Apple Silicon MLX Q4 generation on >=32 GiB machines.
    - "streaming_models_loading": streamed CUDA weights.
    - "full_models_loading": resident CUDA weights.
    """
    if system == "Darwin":
        total_ram = ram_gb if ram_gb is not None else vram_gb
        if macos_major is not None and macos_major < DARWIN_MIN_MACOS_MAJOR:
            return "unsupported"
        if mps_available and total_ram is not None and total_ram >= 32:
            return "mac_mlx_q4"
        return "unsupported"

    if system in ("Windows", "Linux"):
        if not cuda_available:
            return "unsupported"
        if vram_gb is None:
            return "unsupported"
        floor_gb = (
            CUDA_VRAM_FLOOR_GB
            if system == "Windows" and fp8_capable
            else LINUX_VRAM_FLOOR_GB
        )
        if vram_gb < floor_gb:
            return "unsupported"
        # full_models_loading's CUDA_FULL_VRAM_GB floor assumes the fp8-halved (~23 GB)
        # transformer (see module docstring). Without fp8 (ROCm today — see
        # runtime_config.accelerator.accelerator_backend), holding the full bf16
        # (~42-46 GB) transformer resident instead would OOM at this floor, so stay on
        # the streaming path regardless of VRAM until a real bf16-full-resident floor is
        # established on non-CUDA hardware.
        # Originally contributed by boxwrench in https://github.com/Lightricks/LTX-Desktop/pull/160
        if not fp8_capable:
            return "streaming_models_loading"
        if vram_gb < CUDA_FULL_VRAM_GB:
            return "streaming_models_loading"
        return "full_models_loading"

    # Fail closed for non-target platforms unless explicitly relaxed.
    return "unsupported"


def windows_checkpoint_streams_from_disk(
    system: str, drive_type: DriveType | None
) -> bool:
    """Windows + NVMe maps weights from disk; everything else pins from CPU.

    Single source for the DISK path: the offload routing and the profile's
    ``streams_from_disk`` both read this rule, so the RAM floors cannot drift
    from the routing they price.
    """
    return system == "Windows" and drive_type == "nvme"


def should_disk_stream_cuda_weights(
    system: str, ram_gb: int | None, *, drive_type: DriveType | None
) -> bool:
    """Whether CUDA streaming must mmap weights from disk instead of pinning them in RAM.

    Linux actually succeeds at cudaHostRegister, so below the CPU-offload RAM
    floor (or when RAM is unknown) the pinned transformer cannot fit and weights
    stream from disk instead (LTX-Desktop#163). Drive class does not route Linux.
    Windows routes by checkpoint-drive class instead of RAM: NVMe sustains the
    ~70 GB per-generation mapped reads (~20 s); SATA (~130 s) does not earn the
    switch, so non-NVMe and unknown drives stay on OffloadMode.CPU with its
    pageable host buffers (pinned_pool_fix / LTX-Desktop#141).
    """
    if system == "Windows":
        return windows_checkpoint_streams_from_disk(system, drive_type)
    if system != "Linux":
        return False
    if ram_gb is None:
        return True
    return ram_gb < CUDA_CPU_OFFLOAD_RAM_FLOOR_GB


def streaming_prefetch_count_for_mode(mode: LocalGenerationMode) -> int | None:
    """Return the streaming_prefetch_count to pass to a local pipeline.

    Must not be called when local generation is unsupported — callers should
    route through the API instead.
    """
    if mode == "unsupported":
        raise AssertionError(
            "streaming_prefetch_count_for_mode called with 'unsupported' mode; "
            "callers must route to the API instead of constructing a local pipeline."
        )
    if mode in ("full_models_loading", "mac_mlx_q4"):
        return None
    if mode == "streaming_models_loading":
        return 2
    raise AssertionError(f"Unexpected LocalGenerationMode: {mode!r}")
