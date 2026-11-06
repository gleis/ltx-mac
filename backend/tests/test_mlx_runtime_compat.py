from pathlib import Path
import pytest
from services.fast_video_pipeline.prepare_mlx_runtime import pad_audio_mux


def test_mux_padding_is_required_and_idempotent(tmp_path: Path) -> None:
    codec = tmp_path / 'codec.py'
    codec.write_text('cmd.extend(["-i", audio_path, "-c:a", "aac", "-shortest"])\n')
    pad_audio_mux(codec)
    patched = codec.read_text()
    assert '"-af", "apad"' in patched
    assert '"-shortest"' in patched
    pad_audio_mux(codec)
    assert codec.read_text() == patched


def test_mux_padding_refuses_source_drift(tmp_path: Path) -> None:
    codec = tmp_path / 'codec.py'
    codec.write_text('unrecognized future codec')
    with pytest.raises(RuntimeError, match='Unexpected MLX codec source'):
        pad_audio_mux(codec)
    assert codec.read_text() == 'unrecognized future codec'
