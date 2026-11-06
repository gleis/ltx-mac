"""Apply the required, idempotent compatibility fix to pinned MLX 0.14 wheels."""
from __future__ import annotations

from pathlib import Path
import sysconfig

_ORIGINAL = 'cmd.extend(["-i", audio_path, "-c:a", "aac", "-shortest"])'
_PADDED = 'cmd.extend(["-i", audio_path, "-af", "apad", "-c:a", "aac", "-shortest"])'


def pad_audio_mux(path: Path) -> None:
    """Keep the full generated video when the decoded audio is a little shorter.

    apad extends audio with silence; -shortest still stops at the video end.
    Refuse unexpected source drift instead of silently installing a partial fix.
    """
    source = path.read_text()
    if _PADDED in source:
        return
    if source.count(_ORIGINAL) != 1:
        raise RuntimeError(f"Unexpected MLX codec source; cannot safely patch {path}")
    path.write_text(source.replace(_ORIGINAL, _PADDED))


if __name__ == '__main__':
    pad_audio_mux(Path(sysconfig.get_paths()['purelib']) / 'ltx_core_mlx/model/video_vae/video_vae.py')
