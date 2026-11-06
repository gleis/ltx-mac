"""Linux only: open disk-streaming checkpoints with ``pread`` when mmap would hit ENOMEM.

``DiskTensorReader`` keeps a ``safe_open`` handle and ``get_tensor``s each block
during every denoise step (``OffloadMode.DISK``). With no ``backend`` argument
that handle is mmap. safetensors' torch mmap backend calls
``torch.UntypedStorage.from_file(shared=False)`` on the whole file. That is a
private writable mapping, so the kernel charges its full size against the commit
limit. With ``vm.overcommit_memory=0`` the kernel refuses any single mapping
larger than RAM + swap, and with ``=2`` any mapping above the free commit
headroom. A 46 GB checkpoint on a 32 GB / 11 GB-swap host fails with
``unable to mmap 46149345334 bytes ... Cannot allocate memory (12)``
(https://github.com/Lightricks/LTX-Desktop/issues/163). ``pread`` never maps the
file, so it is not charged.

``install`` uses ``pread`` only for a file the kernel would refuse. Every other
Linux open stays on mmap, because Linux DISK ``pread`` throughput is unmeasured.

Isolation: this module is separate from ``disk_streaming_pread`` (the Darwin
patch) on purpose. ``install`` returns before it imports torch, safetensors or
ltx-core unless ``sys.platform`` starts with ``linux``. On macOS and Windows it
touches nothing, and ``disk_streaming_pread`` is the only patch on
``DiskTensorReader``.

Override: ``LTX_LINUX_DISK_STREAM_BACKEND`` is ``auto`` (default, the rule
above), ``mmap`` (never ``pread``; the kernel refusal returns) or ``pread``
(always ``pread``, to benchmark it). Set it where the app starts, for example
``LTX_LINUX_DISK_STREAM_BACKEND=pread ./LTX-Desktop``. Electron passes its
environment to the backend.

Remove once safetensors maps read-only.

Usage:
    import services.patches.linux_disk_streaming_backend as linux_disk_streaming_backend
    linux_disk_streaming_backend.install()
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_BACKEND = "LTX_LINUX_DISK_STREAM_BACKEND"
_BACKEND_CHOICES = ("auto", "mmap", "pread")

_PROC_OVERCOMMIT = Path("/proc/sys/vm/overcommit_memory")
_PROC_MEMINFO = Path("/proc/meminfo")

_installed = False
_checked = False
_original_init: Any = None


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _read_meminfo_bytes() -> dict[str, int]:
    """Parse ``/proc/meminfo`` into bytes. Every value there is in kB."""
    fields: dict[str, int] = {}
    for line in _PROC_MEMINFO.read_text().splitlines():
        key, _, rest = line.partition(":")
        value = rest.split()
        if value:
            fields[key] = int(value[0]) * 1024
    return fields


def mmap_would_be_refused(size_bytes: int) -> bool:
    """Whether the kernel refuses a private writable mapping of ``size_bytes``.

    Mirrors ``__vm_enough_memory``: mode 0 refuses a mapping above RAM + swap,
    mode 2 refuses one above ``CommitLimit - Committed_AS``, mode 1 never refuses.
    When the policy cannot be read, answer False so the open keeps its current
    mmap behaviour.
    """
    try:
        mode = int(_PROC_OVERCOMMIT.read_text())
        meminfo = _read_meminfo_bytes()
        if mode == 1:
            return False
        if mode == 2:
            return size_bytes > meminfo["CommitLimit"] - meminfo["Committed_AS"]
        return size_bytes > meminfo["MemTotal"] + meminfo["SwapTotal"]
    except (OSError, ValueError, KeyError):
        return False


def _backend_choice() -> str:
    """Read ``LTX_LINUX_DISK_STREAM_BACKEND``: ``auto`` (default), ``mmap`` or ``pread``.

    An unknown value warns and acts as ``auto``. It must not stop generation.
    """
    raw = os.environ.get(ENV_BACKEND, "").strip().lower()
    if raw == "":
        return "auto"
    if raw not in _BACKEND_CHOICES:
        logger.warning(
            "Ignoring %s=%r; expected one of %s. Using auto.", ENV_BACKEND, raw, ", ".join(_BACKEND_CHOICES)
        )
        return "auto"
    return raw


def _backend_for(path: str) -> str | None:
    """The safetensors backend to force for ``path``, or None to keep the default."""
    choice = _backend_choice()
    if choice == "mmap":
        return None
    if choice == "pread":
        logger.info("Disk streaming opens %s with backend=pread: %s=pread", path, ENV_BACKEND)
        return "pread"
    size = os.path.getsize(path)
    if mmap_would_be_refused(size):
        logger.info(
            "Disk streaming opens %s (%d bytes) with backend=pread: mmap would be refused "
            "by vm.overcommit_memory",
            path,
            size,
        )
        return "pread"
    return None


def install() -> None:
    """Choose the ``DiskTensorReader`` read backend on Linux. Idempotent.

    No-op on every other platform, before any heavy import. Does not log. This
    runs at import time, before ``logging.basicConfig``, and an info record then
    is dropped. Call ``log_status`` once logging exists.
    """
    global _installed, _checked, _original_init
    if _installed:
        return
    if not _is_linux():
        return

    import safetensors  # noqa: PLC0415

    from ltx_core.block_streaming.disk import DiskTensorReader  # noqa: PLC0415

    _checked = True
    if "backend" not in inspect.signature(safetensors.safe_open).parameters:
        return

    _original_init = DiskTensorReader.__init__

    def _init(self: DiskTensorReader, paths: list[str]) -> None:
        original_open = safetensors.safe_open

        def _open(path: Any, *args: Any, **kwargs: Any) -> Any:
            backend = _backend_for(str(path))
            if backend is not None:
                kwargs.setdefault("backend", backend)
            return original_open(path, *args, **kwargs)

        safetensors.safe_open = _open  # type: ignore[assignment]
        try:
            _original_init(self, paths)
        finally:
            safetensors.safe_open = original_open

    DiskTensorReader.__init__ = _init  # type: ignore[method-assign]
    _installed = True


def log_status() -> None:
    """Report the Linux disk-streaming read backend. Call after logging is configured."""
    if not _checked:
        return
    if not _installed:
        logger.warning("safetensors.safe_open has no backend=; Linux disk streaming stays on mmap")
        return
    choice = _backend_choice()
    if choice == "auto":
        logger.info("Linux disk streaming keeps mmap; pread only when the kernel would refuse the mapping")
    else:
        logger.info("Linux disk streaming backend forced by %s=%s", ENV_BACKEND, choice)
