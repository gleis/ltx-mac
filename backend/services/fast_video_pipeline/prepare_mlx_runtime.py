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


def bound_ffmpeg_diagnostics(path: Path) -> None:
    source = path.read_text()
    if '# LTX23MLX: bounded FFmpeg diagnostic capture' in source:
        return
    old_start = '        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)'
    old_end = '            proc.wait()\n            aggressive_cleanup()'
    if source.count(old_start) != 1 or source.count(old_end) != 1:
        raise RuntimeError('Unexpected MLX FFmpeg process boundary')
    new_start = '        # LTX23MLX: bounded FFmpeg diagnostic capture; never leave a pipe unread.\n        import tempfile as _tempfile\n        import sys as _sys\n        _ffmpeg_errors = _tempfile.TemporaryFile()\n        try:\n            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=_ffmpeg_errors)\n        except Exception:\n            _ffmpeg_errors.close()\n            raise'
    new_end = '            _return_code = proc.wait()\n            _ffmpeg_errors.seek(0)\n            _diagnostics = _ffmpeg_errors.read().decode("utf-8", errors="replace")\n            _ffmpeg_errors.close()\n            aggressive_cleanup()\n            if _return_code != 0 and _sys.exc_info()[0] is None:\n                raise RuntimeError(f"FFmpeg failed ({_return_code}): {_diagnostics[-8192:]}")'
    path.write_text(source.replace(old_start, new_start).replace(old_end, new_end))


def prepare_codec(path: Path) -> None:
    pad_audio_mux(path)
    bound_ffmpeg_diagnostics(path)
    source = path.read_text()
    # Upgrade installed copies of the earlier temporal-only decoder patch.
    if 'PATCHED (LTX23MLX): temporal streaming decode' in source:
        source = source.replace('from ltx_core_mlx.model.video_vae.tiling import TilingConfig, TemporalTilingConfig\n',
                                'from ltx_core_mlx.model.video_vae.tiling import TilingConfig, TemporalTilingConfig, SpatialTilingConfig\n')
        source = source.replace('                    spatial_config=None,',
                                '                    spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=128),')
        path.write_text(source)
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
