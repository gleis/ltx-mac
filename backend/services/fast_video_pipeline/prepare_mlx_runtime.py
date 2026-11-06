"""Apply the required, idempotent compatibility fix to pinned MLX 0.14 wheels."""
from __future__ import annotations

from pathlib import Path
import sysconfig
import runpy

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


def prepare_codec(path: Path) -> None:
    pad_audio_mux(path)
    source = path.read_text()
    anchor = "                weights = _add_at(weights, chunk_coords, mask_slice)"
    evaluated = anchor + "\n                # Bound the lazy graph to one decoded tile.\n                mx.eval(buffer, weights)"
    if evaluated not in source:
        if source.count(anchor) != 1:
            raise RuntimeError("Unexpected MLX tiled decoder; cannot bound tile evaluation")
        path.write_text(source.replace(anchor, evaluated))
    patches = runpy.run_path(str(Path(__file__).with_name('patch_ltx_codec.py')))
    for old, new, marker, label in [
        ('PATCH_CODEC_OLD', 'PATCH_CODEC_NEW', '+faststart', 'output codec'),
        ('PATCH_VAE_STREAM_OLD', 'PATCH_VAE_STREAM_NEW', 'PATCHED (LTX23MLX): temporal streaming decode', 'streaming decoder'),
    ]:
        result = patches['apply_patch'](path, patches[old], patches[new], marker=marker, label=label)
        if result not in (patches['OUTCOME_APPLIED'], patches['OUTCOME_ALREADY']):
            raise RuntimeError(f'Required MLX {label} patch did not apply')


if __name__ == '__main__':
    prepare_codec(Path(sysconfig.get_paths()['purelib']) / 'ltx_core_mlx/model/video_vae/video_vae.py')
