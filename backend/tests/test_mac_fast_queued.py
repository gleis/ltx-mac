"""Queued Home/Remote Fast jobs must apply Mac LTX 2.5 policy before insert."""

from __future__ import annotations

import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import JsonValue
from starlette.testclient import TestClient

from app_factory import create_app
from remote.app import create_remote_app
from runtime_config.video_job_budget import VIDEO_JOB_TOO_LARGE
from services.sqlite_store import SqliteStore
from tests.conftest import TEST_ADMIN_TOKEN
from tests.http_error_assertions import assert_http_error

_GIB = 1024**3
_REMOTE_AUTH = {"Authorization": "Bearer pair-token"}


def _store(test_state) -> SqliteStore:
    return SqliteStore(test_state.config.app_data_dir)


def _configure_mac_fast(
    test_state,
    fake_services,
    *,
    pool_gib: float,
    total_ram_gib: float = 48,
) -> None:
    test_state.config.system = "Darwin"
    test_state.config.total_ram_bytes = int(total_ram_gib * _GIB)
    test_state.config.local_generations_mode = "streaming_models_loading"
    # Deterministic pool: the real probe varies by machine (and is None off-MPS).
    fake_services.gpu_info.mps_pool_bytes = int(pool_gib * _GIB)


@contextmanager
def _surface_client(test_state, surface: str) -> Iterator[tuple[TestClient, dict[str, str]]]:
    if surface == "home":
        app = create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN)
        headers: dict[str, str] = {}
    else:
        app = create_remote_app(handler=test_state, remote_token="pair-token")
        headers = _REMOTE_AUTH
    with TestClient(app) as client:
        yield client, headers


def _t2v_body(*, resolution: str = "540p", duration: int = 5) -> dict[str, JsonValue]:
    return {
        "params": {
            "prompt": "fox",
            "model": "ltx-2.5-fast",
            "aspectRatio": "16:9",
            "resolution": resolution,
            "duration": duration,
            "fps": 24,
        }
    }


def _wav(path: Path, *, duration_seconds: float) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * int(duration_seconds * 8000))
    return path


def _mac_specs(test_state, fake_services, create_fake_model_files, *, pool_gib: float):
    create_fake_model_files()
    _configure_mac_fast(test_state, fake_services, pool_gib=pool_gib)
    app = create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        r = client.get("/api/generate/models-specs")
        assert r.status_code == 200, r.text
        return r.json()["local_models"][0]["spec"]


def test_mac_specs_name_1080p_20s_refusal_with_fence_copy(
    test_state, fake_services, create_fake_model_files
) -> None:
    """1080p/20s is budget-decided, not statically absent: refused with copy.

    124k tokens trip the MPS sequence fence before the pool comparison, so the
    verdict is fixed copy at every pool — pool figures would mislead.
    """
    local_spec = _mac_specs(
        test_state, fake_services, create_fake_model_files, pool_gib=32
    )

    kept_24 = local_spec["supported_resolutions_durations"]["1080p"]["fps_to_durations"]["24"]
    assert 20 not in kept_24
    assert kept_24 == [2, 3, 4, 5, 6, 8, 10]
    for audio in (False, True):
        assert {
            "resolution": "1080p",
            "fps": 24,
            "duration": 20,
            "reason": "mps_sequence_over",
            "message": (
                "This job is too long for this Mac's GPU to run in one pass. "
                "Lower the resolution, duration, frame rate, or aspect width."
            ),
            "audio": audio,
            "over_gib": None,
            "over_ram_gb": None,
            "shortfall_kind": None,
            "readout": None,
            # One pass is refused by the fence: no window is offered instead.
            "windows": None,
            "window_frames": None,
        } in local_spec["dropped_cells"]


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_mac_over_fence_rejects_queued_t2v_before_insert(
    test_state, create_fake_model_files, fake_services, surface: str
) -> None:
    """Mac create-time preflight 422s the MPS-fenced combo; without it the combo queues and SIGABRTs late."""
    create_fake_model_files()
    _configure_mac_fast(test_state, fake_services, pool_gib=32)

    with _surface_client(test_state, surface) as (client, headers):
        response = client.post(
            "/api/generations/text-to-video",
            headers=headers,
            json=_t2v_body(resolution="1080p", duration=20),
        )

    assert_http_error(
        response,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=(
            "This job is too long for this Mac's GPU to run in one pass. "
            "Lower the resolution, duration, frame rate, or aspect width."
        ),
    )
    assert _store(test_state).list_generations("text-to-video") == []


def test_mac_over_pool_rejects_queued_a2v_before_insert(
    test_state, create_fake_model_files, fake_services, tmp_path: Path
) -> None:
    """A2V twin: derived frames at a refusing wider shape never reach the queue."""
    create_fake_model_files()
    _configure_mac_fast(test_state, fake_services, pool_gib=24)

    app = create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        created = client.post(
            "/api/assets",
            json={"path": str(_wav(tmp_path / "line-10s.wav", duration_seconds=10.0))},
        )
        assert created.status_code == 200, created.text
        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": {
                    "prompt": "the character speaks",
                    "model": "ltx-2.5-fast",
                    "resolution": "1080p",
                    "aspectRatio": "21:9",
                    "fps": 24,
                },
                "inputs": {"audio": {"assetId": created.json()["id"]}},
            },
        )

    assert_http_error(
        response,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=(
            "This job needs 30.9 GB of GPU memory but this Mac allows 24.0 GB. "
            "Lower the resolution, duration, frame rate, or aspect width."
        ),
    )
    assert _store(test_state).list_generations("audio-to-video") == []


def test_mac_queued_t2v_admits_cell_under_pool(
    test_state, create_fake_model_files, fake_services
) -> None:
    """The Mac gate refuses only over-pool cells: 720p/20s still queues."""
    create_fake_model_files()
    _configure_mac_fast(test_state, fake_services, pool_gib=32)

    app = create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        response = client.post(
            "/api/generations/text-to-video",
            json=_t2v_body(resolution="720p", duration=20),
        )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"
    queued = _store(test_state).list_generations("text-to-video")
    assert len(queued) == 1
