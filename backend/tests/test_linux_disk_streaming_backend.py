"""Linux disk streaming uses pread only when mmap would hit ENOMEM. Other platforms are untouched."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import safetensors
import torch
from ltx_core.block_streaming.disk import DiskTensorReader
from safetensors.torch import save_file

import services.patches.disk_streaming_pread as darwin_pread
import services.patches.linux_disk_streaming_backend as linux

GIB = 1024**3
KIB_PER_GIB = 1024**2
# A "32 GB" host: the kernel reports ~31 GiB of MemTotal. 31 + 11 GiB swap = 42 GiB < 43 GiB checkpoint.
MEMINFO_32G_HOST = {"MemTotal": 31 * KIB_PER_GIB, "SwapTotal": 11 * KIB_PER_GIB}
CHECKPOINT_46GB = 46_149_345_334  # ltx-2.3-22b-distilled-1.1.safetensors (LTX-Desktop#163)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(linux.ENV_BACKEND, raising=False)
    original_init = DiskTensorReader.__init__
    original_open = safetensors.safe_open
    yield
    DiskTensorReader.__init__ = original_init  # type: ignore[method-assign]
    safetensors.safe_open = original_open  # type: ignore[assignment]
    for module in (linux, darwin_pread):
        module._installed = False
        module._checked = False
        module._original_init = None


def _fake_proc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overcommit: str, meminfo_kib: dict[str, int]
) -> None:
    overcommit_file = tmp_path / "overcommit_memory"
    overcommit_file.write_text(f"{overcommit}\n")
    meminfo_file = tmp_path / "meminfo"
    meminfo_file.write_text("".join(f"{key}:   {kib} kB\n" for key, kib in meminfo_kib.items()))
    monkeypatch.setattr(linux, "_PROC_OVERCOMMIT", overcommit_file)
    monkeypatch.setattr(linux, "_PROC_MEMINFO", meminfo_file)


def _open_with_spy(path: Path) -> list[dict[str, object]]:
    seen: list[dict[str, object]] = []
    original_open = safetensors.safe_open

    def _spy(*args: object, **kwargs: object):
        seen.append(dict(kwargs))
        return original_open(*args, **kwargs)

    safetensors.safe_open = _spy  # type: ignore[assignment]
    try:
        DiskTensorReader([str(path)]).close()
    finally:
        safetensors.safe_open = original_open
    return seen


def _tiny_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.safetensors"
    save_file({"weight": torch.ones(2, 2)}, str(path))
    return path


def _install_on_linux(monkeypatch: pytest.MonkeyPatch, *, refused: bool) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(linux, "mmap_would_be_refused", lambda _size: refused)
    linux.install()


# --- kernel refusal rule ---


def test_refused_when_heuristic_overcommit_and_file_exceeds_ram_plus_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_proc(monkeypatch, tmp_path, "0", MEMINFO_32G_HOST)
    assert linux.mmap_would_be_refused(CHECKPOINT_46GB) is True


def test_allowed_when_file_fits_in_ram_plus_swap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_proc(monkeypatch, tmp_path, "0", MEMINFO_32G_HOST)
    assert linux.mmap_would_be_refused(30 * GIB) is False


def test_allowed_when_overcommit_always(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_proc(monkeypatch, tmp_path, "1", MEMINFO_32G_HOST)
    assert linux.mmap_would_be_refused(CHECKPOINT_46GB) is False


def test_refused_when_strict_overcommit_leaves_too_little_commit_headroom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meminfo = {**MEMINFO_32G_HOST, "CommitLimit": 27 * KIB_PER_GIB, "Committed_AS": 5 * KIB_PER_GIB}
    _fake_proc(monkeypatch, tmp_path, "2", meminfo)
    assert linux.mmap_would_be_refused(23 * GIB) is True
    assert linux.mmap_would_be_refused(21 * GIB) is False


def test_stays_mmap_when_proc_is_unreadable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(linux, "_PROC_OVERCOMMIT", tmp_path / "missing")
    monkeypatch.setattr(linux, "_PROC_MEMINFO", tmp_path / "missing")
    assert linux.mmap_would_be_refused(CHECKPOINT_46GB) is False


# --- DiskTensorReader backend selection on Linux ---


def test_uses_pread_when_mmap_would_be_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _tiny_checkpoint(tmp_path)
    _install_on_linux(monkeypatch, refused=True)
    with caplog.at_level(logging.INFO, logger=linux.logger.name):
        seen = _open_with_spy(path)
    assert seen == [{"framework": "pt", "device": "cpu", "backend": "pread"}]
    assert "mmap would be refused" in caplog.text


def test_keeps_mmap_when_it_fits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _tiny_checkpoint(tmp_path)
    _install_on_linux(monkeypatch, refused=False)
    assert _open_with_spy(path) == [{"framework": "pt", "device": "cpu"}]


def test_pread_reader_returns_the_tensor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _tiny_checkpoint(tmp_path)
    _install_on_linux(monkeypatch, refused=True)
    reader = DiskTensorReader([str(path)])
    try:
        torch.testing.assert_close(reader.get_tensor("weight"), torch.ones(2, 2))
    finally:
        reader.close()


def test_install_is_idempotent_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_on_linux(monkeypatch, refused=False)
    first = DiskTensorReader.__init__
    linux.install()
    assert DiskTensorReader.__init__ is first


def test_install_leaves_mmap_when_backend_is_unsupported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(linux.inspect, "signature", lambda _fn: type("Sig", (), {"parameters": {}})())
    before = DiskTensorReader.__init__
    linux.install()
    assert DiskTensorReader.__init__ is before
    with caplog.at_level(logging.WARNING, logger=linux.logger.name):
        linux.log_status()
    assert "stays on mmap" in caplog.text


# --- override flag ---


def test_flag_pread_forces_pread_when_mmap_would_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _tiny_checkpoint(tmp_path)
    _install_on_linux(monkeypatch, refused=False)
    monkeypatch.setenv(linux.ENV_BACKEND, "pread")
    assert _open_with_spy(path) == [{"framework": "pt", "device": "cpu", "backend": "pread"}]


def test_flag_mmap_forces_mmap_when_kernel_would_refuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _tiny_checkpoint(tmp_path)
    _install_on_linux(monkeypatch, refused=True)
    monkeypatch.setenv(linux.ENV_BACKEND, "mmap")
    assert _open_with_spy(path) == [{"framework": "pt", "device": "cpu"}]


@pytest.mark.parametrize("value", ["auto", "", " AUTO "])
def test_flag_auto_keeps_kernel_driven_choice(
    value: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _tiny_checkpoint(tmp_path)
    _install_on_linux(monkeypatch, refused=True)
    monkeypatch.setenv(linux.ENV_BACKEND, value)
    assert _open_with_spy(path) == [{"framework": "pt", "device": "cpu", "backend": "pread"}]


def test_flag_invalid_value_warns_and_acts_like_auto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _tiny_checkpoint(tmp_path)
    _install_on_linux(monkeypatch, refused=True)
    monkeypatch.setenv(linux.ENV_BACKEND, "preadd")
    with caplog.at_level(logging.WARNING, logger=linux.logger.name):
        seen = _open_with_spy(path)
    assert seen == [{"framework": "pt", "device": "cpu", "backend": "pread"}]
    assert "LTX_LINUX_DISK_STREAM_BACKEND" in caplog.text
    assert "preadd" in caplog.text


def test_log_status_reports_flag_override(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _install_on_linux(monkeypatch, refused=False)
    monkeypatch.setenv(linux.ENV_BACKEND, "pread")
    with caplog.at_level(logging.INFO, logger=linux.logger.name):
        linux.log_status()
    assert "LTX_LINUX_DISK_STREAM_BACKEND=pread" in caplog.text


# --- isolation from macOS and Windows ---


@pytest.mark.parametrize("platform", ["darwin", "win32"])
@pytest.mark.parametrize("flag", [None, "pread", "mmap", "garbage"])
def test_install_touches_nothing_off_linux(
    platform: str, flag: str | None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    if flag is not None:
        monkeypatch.setenv(linux.ENV_BACKEND, flag)
    init_before = DiskTensorReader.__init__
    open_before = safetensors.safe_open
    with caplog.at_level(logging.DEBUG):
        linux.install()
        linux.log_status()
    assert DiskTensorReader.__init__ is init_before
    assert safetensors.safe_open is open_before
    assert linux._installed is False
    assert linux._checked is False
    assert caplog.text == ""


def test_darwin_pread_patch_is_unchanged_when_both_patches_are_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _tiny_checkpoint(tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv(linux.ENV_BACKEND, "mmap")  # Darwin must ignore the Linux flag.
    darwin_pread.install()
    linux.install()
    assert _open_with_spy(path) == [{"framework": "pt", "device": "cpu", "backend": "pread"}]


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_linux_module_does_not_import_ml_stack_off_linux(platform: str) -> None:
    """A fresh interpreter installing on macOS/Windows imports neither safetensors nor ltx_core."""
    import subprocess

    code = (
        "import sys; "
        "import services.patches.linux_disk_streaming_backend as m; "
        "sys.platform = %r; m.install(); m.log_status(); "
        "bad = [n for n in ('safetensors', 'ltx_core') if n in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    ) % platform
    backend_dir = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code], cwd=backend_dir, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
