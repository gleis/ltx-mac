"""Reckless-mode warning tier: setting, gates, annotations, execute downgrade."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
import wave

import pytest
from PIL import Image
from starlette.testclient import TestClient

from _routes._errors import HTTPError
from app_factory import create_app
from handlers.base import resolve_models_dir
from remote.app import create_remote_app
from runtime_config.model_download_specs import resolve_ic_lora_path
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
)
from api_types import (
    AudioToVideoParams,
    GenerateVideoRequest,
    GenerateVideoResponse,
    IcLoraGenerateRequest,
    IcLoraRecipeStoredParams,
    IcLoraSettings,
    LoraEntry,
    LTXLocalModelId,
    VideoAssetMetadata,
    VideoMeta,
)
from runtime_config.ic_lora_job_budget import (
    REFINE_DECODE_FLOOR_MESSAGE,
    TWO_STAGE_OVER_MEASURED_MESSAGE,
    iclora_drop_risk_tier,
)
from runtime_config.ic_lora_tiling import IcLoraTiling
from frame_math import compute_num_frames
from runtime_config.ltx_capabilities import budget_shape, local_caps
from tests.conftest import TEST_ADMIN_TOKEN
from tests.fakes import FakeCapture
from tests.http_error_assertions import assert_http_error
from tests.lora_catalog_helpers import add_ic_lora, download_spec
from services.features.ic_lora_recipes import IC_LORA_RECIPES, IcLoraRecipe
from services.features.video.job_common import spec_risk_allowance, spec_unmeasured_bet
from services.features.video.ic_lora_recipe import IcLoraRecipeExecutor
from services.features.video.text_to_video import TextToVideoExecutor
from services.generation_interrupt import GenerationCancelledError
from services.generation_queue.types import OutputAllocation
from services.records import AssetRecord, GenerationRecord
from services.sqlite_store import SqliteStore
from tests.video_fixtures import write_mp4


def test_settings_round_trip_budget_mode(client: TestClient) -> None:
    # The bug is a missing alias/wire field so the frontend toggle POSTs into
    # the void (extra="forbid" would 422; a missing echo would read back stale).
    posted = client.post("/api/settings", json={"budgetMode": "reckless"})
    assert posted.status_code == 200, posted.text
    fetched = client.get("/api/settings")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["budgetMode"] == "reckless"


def test_local_job_filter_carries_live_budget_mode(test_state) -> None:
    # The bug is specs/preflights rendering with a stale or default mode
    # after the user toggles, because the filter never picked up the setting.
    test_state.state.app_settings.budget_mode = "reckless"
    job_filter = test_state.video_generation._local_job_filter(
        model_id="ltx-2.5-22b-distilled"
    )
    assert job_filter.budget_mode == "reckless"


def test_local_job_filter_defaults_to_standard_mode(test_state) -> None:
    # The bug is a fail-open default that admits over-budget jobs on filters
    # built without an explicit mode (every existing construction site).
    job_filter = test_state.video_generation._local_job_filter(
        model_id="ltx-2.5-22b-distilled"
    )
    assert job_filter.budget_mode == "standard"


_GIB = 1024**3


def _stored_risk(tier: str = "over") -> dict[str, object]:
    return {
        "tier": tier,
        "shortfall_kind": None,
        "over_gib": None,
        "over_ram_gb": None,
        "message": "Over budget.",
    }


def _configure_windows_fast(test_state, *, vram_gib: float, total_ram_gib: float = 48) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = int(vram_gib)
    test_state.config.vram_bytes = int(vram_gib * _GIB)
    test_state.config.total_ram_bytes = int(total_ram_gib * _GIB)
    test_state.config.local_generations_mode = "streaming_models_loading"


def _local_fast_spec(payload: dict, pipeline: str = "fast") -> dict:
    return next(item["spec"] for item in payload["local_models"] if item["pipeline"] == pipeline)


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_specs_echo_live_budget_mode(test_state, create_fake_model_files, surface: str) -> None:
    # The bug is pickers rendering disabled-vs-warned off a stale mode because
    # the specs payload never carries it (Remote has no settings surface).
    create_fake_model_files()
    test_state.state.app_settings.budget_mode = "reckless"
    with _surface_client(test_state, surface) as (client, headers):
        response = client.get("/api/generate/models-specs", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["budgetMode"] == "reckless"


def test_tight_cell_warned_with_headroom_in_both_modes(
    test_state, client: TestClient, create_fake_model_files
) -> None:
    # The bug is a tight fit rendering identical to a comfortable one: no
    # warned cell, so the picker cannot mark it. T1 shows in standard too.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=14.5)
    response = client.get("/api/generate/models-specs")
    assert response.status_code == 200, response.text
    warned = _local_fast_spec(response.json())["warned_cells"]
    tight = [cell for cell in warned if cell["tier"] == "tight"]
    assert tight, f"expected a tight cell, got {warned}"
    cell = next(
        c
        for c in tight
        if c["resolution"] == "1080p" and c["duration"] == 5 and c["fps"] == 24 and c["audio"] is False
    )
    assert cell["headroom_gib"] == pytest.approx(0.36, abs=0.05)
    assert "Tight fit" in cell["message"]


def test_dropped_cell_carries_figures_matching_the_422(
    test_state, client: TestClient, create_fake_model_files
) -> None:
    # The bug is picker "over by X" chrome disagreeing with the dialog because
    # the drop carries prose without the compared figures.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    response = client.get("/api/generate/models-specs")
    assert response.status_code == 200, response.text
    dropped = _local_fast_spec(response.json())["dropped_cells"]
    cell = next(
        c for c in dropped
        if c["resolution"] == "1080p" and c["duration"] == 10 and c["fps"] == 24 and c["audio"] is False
    )
    assert cell["reason"] == "stream_over"
    assert cell["shortfall_kind"] == "total"
    assert cell["over_gib"] == pytest.approx(22.7 - 16.0, abs=0.05)


def test_reckless_empty_kept_grid_with_downgradable_drops_has_no_reason(
    test_state, client: TestClient, create_fake_model_files
) -> None:
    # The bug is reckless rendering the empty-state card instead of the
    # warned form: a ram_shortfall reason survives the mode flip.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16, total_ram_gib=32)
    test_state.state.app_settings.budget_mode = "reckless"
    response = client.get("/api/generate/models-specs")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["local_unavailable_reason"] is None


def test_standard_empty_kept_grid_keeps_ram_reason(
    test_state, client: TestClient, create_fake_model_files
) -> None:
    # Control: the viability change must not leak into standard mode.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16, total_ram_gib=32)
    response = client.get("/api/generate/models-specs")
    assert response.status_code == 200, response.text
    assert response.json()["local_unavailable_reason"] == "ram_shortfall"


def test_iclora_drop_tier_mapping() -> None:
    # The bug is a hard fence (spatial/decode floor) mapped to a bet tier,
    # converting a known hang into an allowed job.
    assert iclora_drop_risk_tier("stream_over") == "over"
    assert iclora_drop_risk_tier("ram_shortfall") == "over"
    assert iclora_drop_risk_tier("two_stage_over_measured") == "unmeasured"
    assert iclora_drop_risk_tier("mac_two_stage_unmeasured") == "unmeasured"
    assert iclora_drop_risk_tier("spatial") is None
    assert iclora_drop_risk_tier("refine_decode_floor") is None
    assert iclora_drop_risk_tier("unsupported") is None


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_iclora_list_echoes_budget_mode(test_state, create_fake_model_files, surface: str) -> None:
    # The bug is IC-LoRA pickers never learning the mode (controls render
    # disabled in reckless because the list payload has no echo).
    test_state.state.app_settings.budget_mode = "reckless"
    with _surface_client(test_state, surface) as (client, headers):
        response = client.get("/api/ic-loras", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["budgetMode"] == "reckless"


def _t2v_body(*, resolution="540p", duration=5, aspect_ratio="16:9", fps=24):
    return {
        "params": {
            "prompt": "fox",
            "model": "ltx-2.5-fast",
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "duration": duration,
            "fps": fps,
        }
    }


@contextmanager
def _surface_client(test_state, surface: str) -> Iterator[tuple[TestClient, dict[str, str]]]:
    if surface == "home":
        app = create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN)
        headers: dict[str, str] = {}
    else:
        app = create_remote_app(handler=test_state, remote_token="pair-token")
        headers = {"Authorization": "Bearer pair-token"}
    with TestClient(app) as client:
        yield client, headers


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_reckless_over_budget_create_enqueues_with_risk_annotation(
    test_state, create_fake_model_files, surface: str
) -> None:
    # The bug is a reckless T2 create 422ing (mode not honored) or enqueueing
    # without the annotation (execute then kills a confirmed bet).
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.state.app_settings.budget_mode = "reckless"
    with _surface_client(test_state, surface) as (client, headers):
        response = client.post(
            "/api/generations/text-to-video",
            headers=headers,
            json=_t2v_body(resolution="1080p", duration=5, aspect_ratio="21:9"),
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["risk"]["tier"] == "over"
    assert "Over budget" in body["risk"]["message"]
    assert body["risk"]["over_gib"] == pytest.approx(17.2 - 16.0, abs=0.05)
    # The annotation survives insert→read (queue badge reads GET, not POST).
    with _surface_client(test_state, surface) as (client, headers):
        fetched = client.get(f"/api/generations/{body['id']}", headers=headers)
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["risk"]["tier"] == "over"


def test_standard_over_budget_create_still_422s_without_insert(
    test_state, client: TestClient, create_fake_model_files
) -> None:
    # Control: reckless must not leak into standard creates.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    response = client.post(
        "/api/generations/text-to-video",
        json=_t2v_body(resolution="1080p", duration=5, aspect_ratio="21:9"),
    )
    assert_http_error(response, status_code=422, code="VIDEO_JOB_TOO_LARGE", message=(
        "This job needs 17.2 GB of GPU memory but this card has 16.0 GB. "
        "Lower the resolution, duration, frame rate, or aspect width."
    ))
    assert test_state.queued_generations.list_generations("text-to-video") == []


def test_reckless_unsupported_create_still_422s(
    test_state, client: TestClient, create_fake_model_files
) -> None:
    # The bug is reckless annotating a missing pipeline (no bet exists when
    # nothing can run it). Flip the process mode: preflight must still refuse.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.config.local_generations_mode = "unsupported"
    test_state.state.app_settings.budget_mode = "reckless"
    response = client.post(
        "/api/generations/text-to-video",
        json=_t2v_body(resolution="540p", duration=5),
    )
    assert_http_error(response, status_code=422, code="LOCAL_GENERATION_UNSUPPORTED", message=(
        LOCAL_GENERATION_UNSUPPORTED_MESSAGE
    ))


def test_downgraded_job_streams_never_full(
    test_state, fake_services, create_fake_model_files
) -> None:
    # The bug is an over-budget job picking full and paging to death: the
    # downgrade must pin streaming.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.state.app_settings.budget_mode = "reckless"
    job_filter = test_state.video_generation._local_job_filter(
        model_id="ltx-2.5-22b-distilled", total_only=True
    )
    mode, risk = test_state.video_generation._video_job_load_mode(
        2560, 1088, 121, job_filter=job_filter
    )
    assert mode == "streaming_models_loading"
    assert risk is not None and risk.tier == "over"


def test_evict_win_beats_risk_annotation(
    test_state, fake_services, create_fake_model_files
) -> None:
    # The bug is reckless recording a bet (and warning the user) for a job a
    # free eviction would have won. W-band shape (720p/20s asks 20.03 on a
    # 24 GB card): occupied free 19.0 rejects, evicting the warm weights frees
    # to 23.5, so the re-decide loads and the annotation stays None.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=24)
    test_state.state.app_settings.budget_mode = "reckless"
    width, height = budget_shape(
        local_caps("ltx-2.5-22b-distilled"), "720p", "16:9", mode="video"
    )
    frames = compute_num_frames(20, 24)
    samples = iter([19.0, 23.5])
    fake_services.gpu_info.free_vram_bytes_fn = lambda: int(next(samples, 23.5) * _GIB)
    fake_services.gpu_info.live_allocated_bytes = int(4.5 * _GIB)
    job_filter = test_state.video_generation._local_job_filter(
        model_id="ltx-2.5-22b-distilled"
    )
    mode, risk = test_state.video_generation._video_job_load_mode(
        width, height, frames, job_filter=job_filter, allow_evict_retry=True
    )
    assert mode == "streaming_models_loading"
    assert risk is None


# --- Task 7: edit + IC-LoRA create-time annotations ---


def _configure_windows_edit(test_state, *, vram_gib: float, total_ram_gib: float) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = int(vram_gib)
    test_state.config.vram_bytes = int(vram_gib * _GIB)
    test_state.config.total_ram_bytes = int(total_ram_gib * _GIB)
    test_state.config.local_generations_mode = "streaming_models_loading"


def _configure_mac_iclora(test_state, fake_services, *, pool_gib: float) -> None:
    test_state.config.system = "Darwin"
    test_state.config.total_ram_bytes = int(48 * _GIB)
    test_state.config.local_generations_mode = "streaming_models_loading"
    # Deterministic pool: the real probe varies by machine (and is None off-MPS).
    fake_services.gpu_info.mps_pool_bytes = int(pool_gib * _GIB)


def _retake_body(asset_id: str, *, start: float = 0.0, duration: float = 2.0) -> dict:
    return {
        "params": {
            "prompt": "the character turns",
            "model": "ltx-2.5-fast",
            "startTime": start,
            "duration": duration,
            "mode": "replace_video",
        },
        "inputs": {"video": {"assetId": asset_id}},
    }


def _upload_video(home_client: TestClient, path: Path) -> dict:
    uploaded = home_client.post("/api/assets", json={"path": str(path)})
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()


def _wav(path: Path, *, duration_seconds: float = 8.0) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * int(duration_seconds * 8000))
    return path


def _recipe_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "prompt": "turn the street to night",
        "model": "ltx-2.5-fast",
        "resolution": "540p",
        "audioMode": "generated",
        "variantId": "default",
    }
    params.update(overrides)
    return params


def _install_layout_catalog(fake_services) -> None:
    """The Layout to Render shape: stage 2 keeps the IC-LoRA, no extra controls."""
    add_ic_lora(
        fake_services,
        id="layout-to-render",
        download=download_spec("layout.safetensors"),
        input={"kind": "video"},
        allows_reference_image=True,
        reference_image_required=True,
        reference_image_frame=-1,
        default_settings=IcLoraSettings(
            skip_stage_2=False, stage_2_ic_lora=True, resolution_factor=1.0, audio_mode="source"
        ),
    )


def _write_recipe_weights(test_state, ic_lora_id: str, filename: str = "layout.safetensors") -> None:
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    path = resolve_ic_lora_path(models_dir, ic_lora_id, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")


def _install_refine_catalog(fake_services) -> None:
    """A Refine-shaped tiled entry: single-stage, never chunks, so a long
    high-resolution clip trips the decoder floor instead of pricing."""
    add_ic_lora(
        fake_services,
        id="probe-refine",
        download=download_spec("refine.safetensors"),
        input={"kind": "video"},
        default_settings=IcLoraSettings(
            skip_stage_2=True,
            resolution_factor=1.5,
            audio_mode="source",
            tiling=IcLoraTiling(long_side=1024, short_side=576),
        ),
    )


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_reckless_over_budget_retake_enqueues_with_risk(
    test_state, client: TestClient, create_fake_model_files, surface: str, tmp_path: Path
) -> None:
    # The bug is a reckless edit create 422ing while Fast creates annotate
    # (one feature's mode wiring missed), or annotating without figures.
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=8, total_ram_gib=64)
    test_state.state.app_settings.budget_mode = "reckless"
    clip = write_mp4(tmp_path / "clip.mp4", seconds=4, fps=24, width=1920, height=1080)
    asset_id = _upload_video(client, clip)["id"]
    with _surface_client(test_state, surface) as (surface_client, headers):
        response = surface_client.post(
            "/api/generations/retake",
            headers=headers,
            json=_retake_body(asset_id, start=0.0, duration=2.0),
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["risk"]["tier"] == "over"
    assert body["risk"]["over_gib"] == pytest.approx(11.8 - 8.0, abs=0.15)
    # The annotation survives insert→read (queue badge reads GET, not POST).
    with _surface_client(test_state, surface) as (surface_client, headers):
        fetched = surface_client.get(f"/api/generations/{body['id']}", headers=headers)
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["risk"]["tier"] == "over"


def test_reckless_overlong_selection_still_422s(
    test_state, client: TestClient, create_fake_model_files, tmp_path: Path
) -> None:
    # The bug is reckless waiving window math (product rules, not VRAM):
    # an over-long selection must refuse in every mode.
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    test_state.state.app_settings.budget_mode = "reckless"
    clip = write_mp4(tmp_path / "clip.mp4", seconds=241 / 24, fps=24, width=1920, height=1080)
    asset_id = _upload_video(client, clip)["id"]
    with _surface_client(test_state, "home") as (surface_client, headers):
        response = surface_client.post(
            "/api/generations/retake",
            headers=headers,
            json=_retake_body(asset_id, start=0.0, duration=4.0),
        )
    assert_http_error(
        response,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=(
            "Retake selection of 4s is too long for 1920x1080 on this machine "
            "(at most 3s)."
        ),
    )
    assert SqliteStore(test_state.config.app_data_dir).list_generations("retake") == []


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_reckless_unmeasured_iclora_create_enqueues_with_risk(
    test_state, client: TestClient, create_fake_model_files, fake_services, surface: str, tmp_path: Path
) -> None:
    # The bug is a T3 IC-LoRA config (unmeasured two-stage length) 422ing in
    # reckless, or annotating as "over" with fabricated figures.
    create_fake_model_files()
    _configure_mac_iclora(test_state, fake_services, pool_gib=64)
    _install_layout_catalog(fake_services)
    _write_recipe_weights(test_state, "layout-to-render")
    test_state.state.app_settings.budget_mode = "reckless"
    clip = write_mp4(tmp_path / "clip.mp4", seconds=10, fps=24, width=1920, height=1088)
    video = _upload_video(client, clip)
    test_state.video_processor.register_video(
        video["path"], FakeCapture(frames=["f"] * 240, fps=24, width=1920, height=1088)
    )
    look_path = tmp_path / "look.png"
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(look_path, format="PNG")
    image = _upload_video(client, look_path)
    with _surface_client(test_state, surface) as (surface_client, headers):
        response = surface_client.post(
            "/api/generations/ic-lora-recipes/layout-to-render",
            headers=headers,
            json={
                "params": _recipe_params(resolution="1080p", fps=24),
                "inputs": {"video": {"assetId": video["id"]}, "image": {"assetId": image["id"]}},
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["risk"]["tier"] == "unmeasured"
    assert body["risk"]["over_gib"] is None
    assert body["risk"]["over_ram_gb"] is None
    # The annotation survives insert→read (queue badge reads GET, not POST).
    with _surface_client(test_state, surface) as (surface_client, headers):
        fetched = surface_client.get(f"/api/generations/{body['id']}", headers=headers)
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["risk"]["tier"] == "unmeasured"


def test_deleted_t3_branch_regresses_windows_two_stage_create_to_422(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # The bug is a deleted or mis-mapped two_stage_over_measured branch in
    # _risk_for_iclora_verdict regressing Windows T3 creates to 422: standard
    # must still refuse with the fence copy, reckless must annotate unmeasured
    # with no figures. Pinned at direct preflight because no recipe POST can
    # reach it — the only never-chunking two-stage layout (STAGE_1_LORA) fails
    # the home-recipe gate before the budget gate.
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=32, total_ram_gib=64)
    item = add_ic_lora(
        fake_services,
        id="probe-ts",
        input={"kind": "video"},
        default_settings=IcLoraSettings(
            skip_stage_2=False, resolution_factor=1.0, audio_mode="source"
        ),
    )
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(path), FakeCapture(frames=["f"] * 249, fps=24, width=1920, height=1080)
    )
    video = AssetRecord(
        id="v1",
        media_kind="video",
        origin="uploaded",
        path=str(path),
        mime_type="video/mp4",
        name="clip.mp4",
        metadata=VideoAssetMetadata(
            mediaType="video",
            metadata=VideoMeta(
                width=1920,
                height=1080,
                durationMs=10375,
                sizeBytes=100,
                audioStreamCount=0,
                fps=24.0,
            ),
        ),
        created_at=0,
    )
    stored = IcLoraRecipeStoredParams(
        prompt="p",
        model="ltx-2.5-fast",
        resolution="1080p",
        audioMode="generated",
        fps=24,
        loras=[LoraEntry(ref="", scale=1.0, catalogId="probe-ts")],
    )
    with pytest.raises(HTTPError) as exc_info:
        test_state.ic_lora.preflight_queued_recipe_job(
            item=item, video=video, stored=stored, source_fps=24.0
        )
    assert exc_info.value.status_code == 422
    assert exc_info.value.code == VIDEO_JOB_TOO_LARGE
    assert exc_info.value.detail == TWO_STAGE_OVER_MEASURED_MESSAGE
    test_state.state.app_settings.budget_mode = "reckless"
    risk = test_state.ic_lora.preflight_queued_recipe_job(
        item=item, video=video, stored=stored, source_fps=24.0
    )
    assert risk is not None
    assert risk.tier == "unmeasured"
    assert risk.over_gib is None
    assert risk.over_ram_gb is None


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_reckless_decode_floor_iclora_create_still_422s(
    test_state, client: TestClient, create_fake_model_files, fake_services, surface: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Substitutes the planned test_reckless_spatial_iclora_create_still_422s:
    # IC_LORA_SOURCE_TOO_LARGE is unreachable through recipe creates — inputs
    # clamp via fit_ic_lora_stage1
    # (backend/runtime_config/ic_lora_local_envelope.py:123) / fit_in_box
    # (backend/runtime_config/grid_fit.py:10) before the verdict — and both
    # reasons share the same hard-wall `return None` in _risk_for_iclora_verdict
    # (backend/handlers/ic_lora_handler.py:408), so this test pins the branch
    # the spatial case would have pinned.
    # The bug is reckless waiving a pipeline limit (a known decode failure,
    # not a bet): a tiled Refine job past the one-pass shape must refuse in
    # every mode.
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=32, total_ram_gib=64)
    _install_refine_catalog(fake_services)
    monkeypatch.setitem(
        IC_LORA_RECIPES,
        "probe-refine",
        IcLoraRecipe(recipe_id="probe-refine", catalog_id="probe-refine"),
    )
    _write_recipe_weights(test_state, "probe-refine", "refine.safetensors")
    test_state.state.app_settings.budget_mode = "reckless"
    clip = write_mp4(tmp_path / "clip.mp4", seconds=10, fps=24, width=1920, height=1080)
    video = _upload_video(client, clip)
    test_state.video_processor.register_video(
        video["path"], FakeCapture(frames=["f"] * 240, fps=24, width=1920, height=1080)
    )
    with _surface_client(test_state, surface) as (surface_client, headers):
        response = surface_client.post(
            "/api/generations/ic-lora-recipes/probe-refine",
            headers=headers,
            json={
                "params": _recipe_params(resolution="1080p", fps=24),
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )
    assert_http_error(
        response,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=REFINE_DECODE_FLOOR_MESSAGE,
    )
    assert SqliteStore(test_state.config.app_data_dir).list_generations("probe-refine") == []


# --- Task 8: execute downgrade keyed off the stored annotation ---


def test_allow_risk_none_follows_live_mode(test_state, create_fake_model_files) -> None:
    # The bug is request paths ignoring the toggle (hardcoded False) or
    # executors inheriting it (None passed through).
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    standard_filter = test_state.video_generation._local_job_filter(
        model_id="ltx-2.5-22b-distilled", total_only=True
    )
    # standard mode + None ⇒ raises
    with pytest.raises(HTTPError):
        test_state.video_generation._video_job_load_mode(
            2560, 1088, 121, job_filter=standard_filter
        )
    test_state.state.app_settings.budget_mode = "reckless"
    live_filter = test_state.video_generation._local_job_filter(
        model_id="ltx-2.5-22b-distilled", total_only=True
    )
    mode, risk = test_state.video_generation._video_job_load_mode(
        2560, 1088, 121, job_filter=live_filter
    )
    assert mode == "streaming_models_loading"
    assert risk is not None and risk.tier == "over"


def test_explicit_allow_risk_false_overrides_reckless_mode(
    test_state, create_fake_model_files
) -> None:
    # The bug is an unconfirmed drifted job running because the toggle moved:
    # executors pass explicit False for unannotated jobs in reckless mode.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.state.app_settings.budget_mode = "reckless"
    reckless_filter = test_state.video_generation._local_job_filter(
        model_id="ltx-2.5-22b-distilled", total_only=True
    )
    with pytest.raises(HTTPError) as exc_info:
        test_state.video_generation._video_job_load_mode(
            2560, 1088, 121, job_filter=reckless_filter, allow_risk=False
        )
    assert exc_info.value.status_code == 422


def test_explicit_allow_risk_true_survives_standard_mode(
    test_state, create_fake_model_files
) -> None:
    # The bug is a confirmed bet dying because the toggle moved: stored
    # annotation ⇒ explicit True ⇒ downgrades on a standard filter.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    standard_filter = test_state.video_generation._local_job_filter(
        model_id="ltx-2.5-22b-distilled", total_only=True
    )
    mode, risk = test_state.video_generation._video_job_load_mode(
        2560, 1088, 121, job_filter=standard_filter, allow_risk=True
    )
    assert mode == "streaming_models_loading"
    assert risk is not None and risk.tier == "over"


def test_allow_risk_never_waives_hard_reject(
    test_state, create_fake_model_files
) -> None:
    # The bug is a stale annotation waiving a missing pipeline at execute
    # (e.g. model uninstalled between queue and run).
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.config.local_generations_mode = "unsupported"
    unsupported_filter = test_state.video_generation._local_job_filter(
        model_id="ltx-2.5-22b-distilled", total_only=True
    )
    with pytest.raises(HTTPError):
        test_state.video_generation._video_job_load_mode(
            2560, 1088, 121, job_filter=unsupported_filter, allow_risk=True
        )


class _CaptureGenerator:
    """ReservedVideoGenerator fake that records the risk kwarg, then stops before GPU work."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.seen: dict[str, object] = {}

    def validate_local_a2v_request(
        self, params: AudioToVideoParams, audio_duration_ms: int
    ) -> None:
        del params, audio_duration_ms

    def generate_local_reserved(
        self,
        req: GenerateVideoRequest,
        *,
        generation_id: str,
        output_path: Path | None = None,
        prompt_wrap: Callable[[str], str] | None = None,
        a2v_num_frames: int | None = None,
        a2v_audio_duration_seconds: float | None = None,
        local_model_id: LTXLocalModelId,
        risk_allowance: bool | None = None,
        unmeasured_bet: bool = False,
    ) -> GenerateVideoResponse:
        del (
            req,
            generation_id,
            output_path,
            prompt_wrap,
            a2v_num_frames,
            a2v_audio_duration_seconds,
            local_model_id,
            unmeasured_bet,
        )
        self.seen = {"risk_allowance": risk_allowance}
        raise GenerationCancelledError()


class _CaptureIcLoraGenerator:
    """ReservedIcLoraGenerator fake that records the risk kwarg, then stops before GPU work."""

    def __init__(self, ic_lora) -> None:
        self._ic_lora = ic_lora
        self.seen: dict[str, object] = {}

    def recipe_run_frames(self, **kwargs: object):
        return self._ic_lora.recipe_run_frames(**kwargs)

    def generate_local_reserved(
        self,
        req: IcLoraGenerateRequest,
        *,
        generation_id: str,
        output_path: Path,
        local_model_id: LTXLocalModelId,
        seed: int,
        explore_generation: bool = True,
        explicit_generation_seed: bool = False,
        high_res: bool = False,
        risk_allowance: bool | None = None,
        unmeasured_bet: bool = False,
    ) -> object:
        del (
            req,
            generation_id,
            output_path,
            local_model_id,
            seed,
            explore_generation,
            explicit_generation_seed,
            high_res,
        )
        self.seen = {
            "risk_allowance": risk_allowance,
            "unmeasured_bet": unmeasured_bet,
        }
        raise GenerationCancelledError()


class _StillAssets:
    """AssetLookup fake serving one prebuilt record."""

    def __init__(self, asset: AssetRecord) -> None:
        self._asset = asset

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        return self._asset if asset_id == self._asset.id else None


def test_executor_passes_stored_allowance_to_generator(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    # The bug is the executor dropping the annotation on the floor: generate
    # never receives the allowance and the confirmed bet dies at execute.
    assert spec_risk_allowance({"params": {}, "_risk": _stored_risk("over")}) is True
    assert spec_risk_allowance({"params": {}, "_risk": _stored_risk("unmeasured")}) is False
    assert spec_unmeasured_bet({"params": {}, "_risk": _stored_risk("unmeasured")}) is True
    assert spec_unmeasured_bet({"params": {}, "_risk": _stored_risk("over")}) is False
    assert spec_risk_allowance({"params": {}}) is False
    assert spec_risk_allowance({"params": {}, "_risk": {"tier": "tight", "message": "x"}}) is False
    assert spec_risk_allowance({"params": {}, "_risk": "junk"}) is False
    assert spec_risk_allowance({"params": {}, "_risk": {"tier": "over"}}) is False
    # And the executor threads it: capture the kwarg with a tiny Protocol fake.
    create_fake_model_files()
    generator = _CaptureGenerator(test_state.config.default_models_dir)
    executor = TextToVideoExecutor(generator)
    generation = GenerationRecord(
        id="t2v-risk",
        feature="text-to-video",
        contract_version=1,
        status="running",
        spec={
            "params": {
                "prompt": "a red fox",
                "resolution": "540p",
                "model": "ltx-2.5-fast",
                "duration": 8,
                "fps": 24,
                "seed": 12345,
            },
            "_risk": _stored_risk(),
        },
        error_code=None,
        created_at=1,
        queued_at=1,
        attempt_count=1,
        started_at=1,
        finished_at=None,
        outputs=(),
    )
    plans = executor.plan_outputs(generation)
    outputs = (
        OutputAllocation(
            plan=plans[0], asset_id="asset-1", dest_path=str(tmp_path / "output.mp4")
        ),
    )
    with pytest.raises(GenerationCancelledError):
        executor.execute(generation, outputs)
    assert generator.seen == {"risk_allowance": True}


def test_retry_preserves_risk_annotation(
    test_state, client: TestClient, create_fake_model_files
) -> None:
    # The bug is a retry dropping _risk: the same confirmed bet dies at
    # execute the second time around.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.state.app_settings.budget_mode = "reckless"
    # Hold the GPU slot so the app's queue runner cannot claim the row first.
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        response = client.post(
            "/api/generations/text-to-video",
            json=_t2v_body(resolution="1080p", duration=5, aspect_ratio="21:9"),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["risk"]["tier"] == "over"
        store = SqliteStore(test_state.config.app_data_dir)
        claimed = store.claim_next_queued()
        assert claimed is not None
        store.mark_failed(
            body["id"], "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count
        )
        retried = client.post(f"/api/generations/{body['id']}/retry")
    finally:
        test_state.generation.release_generation_start_reservation()
    assert retried.status_code == 200, retried.text
    assert retried.json()["risk"]["tier"] == "over"


def test_corrupt_stored_risk_does_not_fail_generation_reads(
    test_state, client: TestClient
) -> None:
    # The bug is one junk _risk row raising on list and get, which also
    # covers queue, retry, and cancel: they all read through the same helper.
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        store = SqliteStore(test_state.config.app_data_dir)
        junk = store.insert_generation(
            "text-to-video",
            {"params": {"prompt": "junk"}, "_risk": {"tier": "over"}},
        )
        valid = store.insert_generation(
            "text-to-video",
            {"params": {"prompt": "valid"}, "_risk": _stored_risk()},
        )
        listed = client.get("/api/generations", params={"feature": "text-to-video"})
        assert listed.status_code == 200, listed.text
        by_id = {row["id"]: row for row in listed.json()}
        assert by_id[junk.id]["risk"] is None
        assert by_id[valid.id]["risk"]["tier"] == "over"
        fetched = client.get(f"/api/generations/{junk.id}")
        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["risk"] is None
    finally:
        test_state.generation.release_generation_start_reservation()


def test_estimate_returns_risk_not_422_in_reckless(
    test_state, client: TestClient, create_fake_model_files
) -> None:
    # The bug is estimate 422ing a selection the form allows: any UI that
    # prices before generating needs the annotation, not the refusal.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.state.app_settings.budget_mode = "reckless"
    response = client.post(
        "/api/generate/estimate",
        json={"resolution": "1080p", "aspectRatio": "21:9", "duration": 5, "fps": 24},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["risk"]["tier"] == "over"
    assert "estimate" in body  # ETA or null, never an error


def test_estimate_still_422s_in_standard(
    test_state, client: TestClient, create_fake_model_files
) -> None:
    # Control: estimate keeps today's 422s in standard mode.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    response = client.post(
        "/api/generate/estimate",
        json={"resolution": "1080p", "aspectRatio": "21:9", "duration": 5, "fps": 24},
    )
    assert_http_error(response, status_code=422, code="VIDEO_JOB_TOO_LARGE", message=(
        "This job needs 17.2 GB of GPU memory but this card has 16.0 GB. "
        "Lower the resolution, duration, frame rate, or aspect width."
    ))


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_queue_snapshot_carries_risk_on_both_surfaces(
    test_state, create_fake_model_files, surface: str
) -> None:
    # The bug is the queue badge starving: the annotation exists on the row
    # but the snapshot (or the Remote projection) drops it.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.state.app_settings.budget_mode = "reckless"
    with _surface_client(test_state, surface) as (client, headers):
        created = client.post(
            "/api/generations/text-to-video",
            headers=headers,
            json=_t2v_body(resolution="1080p", duration=5, aspect_ratio="21:9"),
        )
        assert created.status_code == 200, created.text
        snapshot = client.get("/api/generation-queue", headers=headers)
    assert snapshot.status_code == 200, snapshot.text
    body = snapshot.json()
    entries = list(body["queued"])
    if body["active"] is not None:
        entries.append(body["active"])
    entries.extend(body["done"])
    entries.extend(body["failed"])
    risky = [e for e in entries if e["generation"]["id"] == created.json()["id"]]
    assert risky and risky[0]["generation"]["risk"]["tier"] == "over"


def test_reckless_dropped_a2v_row_reports_budget_not_invalid(
    test_state, client: TestClient, create_fake_model_files, tmp_path: Path
) -> None:
    # The bug is reckless swallowing the derive-time budget refusal (the
    # downgrade path returns a risk instead of raising), so a fully-dropped
    # A2V row 422s INVALID instead of the mode-stable VIDEO_JOB_TOO_LARGE.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=12)
    test_state.state.app_settings.budget_mode = "reckless"
    audio = client.post(
        "/api/assets",
        json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=5.0))},
    ).json()
    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": {
                "prompt": "the character speaks",
                "resolution": "1080p",
                "model": "ltx-2.5-fast",
                "fps": 24,
            },
            "inputs": {"audio": {"assetId": audio["id"]}},
        },
    )
    assert response.status_code == 422, response.text
    assert_http_error(response, status_code=422, code="VIDEO_JOB_TOO_LARGE", message=(
        "This job needs 13.6 GB of GPU memory but this card has 12.0 GB. "
        "Lower the resolution, duration, frame rate, or aspect width."
    ))
    assert SqliteStore(test_state.config.app_data_dir).list_generations("audio-to-video") == []


@pytest.mark.parametrize("budget_mode", ["standard", "reckless"])
def test_fully_dropped_a2v_row_admits_audio_shorter_than_every_cell(
    test_state, client: TestClient, create_fake_model_files, tmp_path: Path, budget_mode: str
) -> None:
    # 4s audio loads on 12 GB 1080p after every cell was dropped; both modes must 422 that cell, not enqueue.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=12)
    test_state.state.app_settings.budget_mode = budget_mode
    specs = client.get("/api/generate/models-specs")
    assert specs.status_code == 200, specs.text
    spec = _local_fast_spec(specs.json())
    assert "1080p" not in spec["a2v_supported_resolutions_durations"]
    dropped = [
        cell
        for cell in spec["dropped_cells"]
        if cell["resolution"] == "1080p" and cell["fps"] == 24 and cell["audio"] is True
    ]
    shortest = min(dropped, key=lambda cell: cell["duration"])
    assert shortest["duration"] == 5
    audio = client.post(
        "/api/assets",
        json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=4.0))},
    ).json()
    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": {
                "prompt": "the character speaks",
                "resolution": "1080p",
                "model": "ltx-2.5-fast",
                "fps": 24,
            },
            "inputs": {"audio": {"assetId": audio["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=422,
        code="VIDEO_JOB_TOO_LARGE",
        message=shortest["message"],
    )
    assert SqliteStore(test_state.config.app_data_dir).list_generations("audio-to-video") == []


def test_partial_a2v_drop_refuses_the_static_duration_not_invalid_audio(
    test_state, client: TestClient, create_fake_model_files, tmp_path: Path
) -> None:
    # 1080p keeps 5–6s at 16 GB and drops 10s. 10s audio is inside the static
    # ceiling, so the refusal is the budget, not "audio is too long".
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    audio = client.post(
        "/api/assets",
        json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=10.0))},
    ).json()
    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": {
                "prompt": "the character speaks",
                "resolution": "1080p",
                "model": "ltx-2.5-fast",
                "fps": 24,
            },
            "inputs": {"audio": {"assetId": audio["id"]}},
        },
    )
    assert response.status_code == 422, response.text
    assert_http_error(response, status_code=422, code="VIDEO_JOB_TOO_LARGE", message=(
        "This job needs 22.1 GB of GPU memory but this card has 16.0 GB. "
        "Lower the resolution, duration, frame rate, or aspect width."
    ))
    assert SqliteStore(test_state.config.app_data_dir).list_generations("audio-to-video") == []


def test_reckless_partial_a2v_drop_enqueues_the_static_duration(
    test_state, client: TestClient, create_fake_model_files, tmp_path: Path
) -> None:
    # Same dropped 10s cell. Reckless must enqueue the static 10s shape
    # (233 frames), not trim it to the kept 6s cell (137 frames).
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.state.app_settings.budget_mode = "reckless"
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        audio = client.post(
            "/api/assets",
            json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=10.0))},
        ).json()
        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": {
                    "prompt": "the character speaks",
                    "resolution": "1080p",
                    "model": "ltx-2.5-fast",
                    "fps": 24,
                },
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )
    finally:
        test_state.generation.release_generation_start_reservation()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["spec"]["params"]["numFrames"] == 233
    assert body["risk"]["tier"] == "over"
    assert body["risk"]["over_gib"] is not None


def _extend_body(asset_id: str, *, duration: float = 2.0, mode: str = "end") -> dict:
    return {
        "params": {
            "prompt": "the scene continues",
            "model": "ltx-2.5-fast",
            "duration": duration,
            "mode": mode,
        },
        "inputs": {"video": {"assetId": asset_id}},
    }


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_reckless_over_budget_extend_enqueues_with_risk(
    test_state, client: TestClient, create_fake_model_files, surface: str, tmp_path: Path
) -> None:
    # The bug is a reckless extend create 422ing while retake creates annotate
    # (one feature's mode wiring missed), or annotating without figures.
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=8, total_ram_gib=64)
    test_state.state.app_settings.budget_mode = "reckless"
    clip = write_mp4(tmp_path / "clip.mp4", seconds=4, fps=24, width=1920, height=1080)
    asset_id = _upload_video(client, clip)["id"]
    with _surface_client(test_state, surface) as (surface_client, headers):
        response = surface_client.post(
            "/api/generations/extend",
            headers=headers,
            json=_extend_body(asset_id, duration=2.0),
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["risk"]["tier"] == "over"
    assert body["risk"]["over_gib"] == pytest.approx(15.2 - 8.0, abs=0.15)
    # The annotation survives insert→read (queue badge reads GET, not POST).
    with _surface_client(test_state, surface) as (surface_client, headers):
        fetched = surface_client.get(f"/api/generations/{body['id']}", headers=headers)
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["risk"]["tier"] == "over"


def test_iclora_executor_passes_stored_allowance_to_generator(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # The bug is the IC-LoRA recipe executor dropping the annotation on the
    # floor: the generator re-gates on the live mode and the confirmed bet
    # dies at execute after a flip.
    create_fake_model_files()
    add_ic_lora(fake_services, id="day-to-night")
    source = write_mp4(tmp_path / "phone.mp4", seconds=1, fps=30, width=64, height=64)
    asset = AssetRecord(
        id="clip",
        media_kind="video",
        origin="uploaded",
        path=str(source),
        mime_type="video/mp4",
        name="phone.mp4",
        metadata=VideoAssetMetadata(
            mediaType="video",
            metadata=VideoMeta(
                width=64,
                height=64,
                durationMs=1000,
                sizeBytes=source.stat().st_size,
                audioStreamCount=0,
                fps=30.0,
            ),
        ),
        created_at=0,
    )
    generator = _CaptureIcLoraGenerator(test_state.ic_lora)
    executor = IcLoraRecipeExecutor(
        generator,
        _StillAssets(asset),
        lambda: test_state.config.default_models_dir,
        catalog_id="day-to-night",
    )
    stored = IcLoraRecipeStoredParams(
        prompt="night",
        model="ltx-2.5-fast",
        resolution="540p",
        audioMode="off",
        fps=25,
        loras=[LoraEntry(ref="", scale=1.0, catalogId="day-to-night")],
    ).model_dump(mode="json")

    def _run(risk: dict[str, object] | None) -> None:
        spec: dict[str, object] = {
            "params": stored,
            "inputs": {"video": {"assetId": asset.id}},
        }
        if risk is not None:
            spec["_risk"] = risk
        generation = GenerationRecord(
            id="iclora-risk",
            feature="day-to-night",
            contract_version=1,
            status="running",
            spec=spec,
            error_code=None,
            created_at=1,
            queued_at=1,
            attempt_count=1,
            started_at=1,
            finished_at=None,
            outputs=(),
        )
        plans = executor.plan_outputs(generation)
        outputs = (
            OutputAllocation(
                plan=plans[0], asset_id="asset-1", dest_path=str(tmp_path / "output.mp4")
            ),
        )
        with pytest.raises(GenerationCancelledError):
            executor.execute(generation, outputs)

    _run(_stored_risk("unmeasured"))
    assert generator.seen == {"risk_allowance": False, "unmeasured_bet": True}
    _run(_stored_risk("over"))
    assert generator.seen == {"risk_allowance": True, "unmeasured_bet": False}
    _run(None)
    assert generator.seen == {"risk_allowance": False, "unmeasured_bet": False}


def test_edit_preflight_echoes_live_budget_mode(
    test_state, client: TestClient, create_fake_model_files, tmp_path: Path
) -> None:
    # The bug is edit pickers rendering disabled-vs-warned off a stale mode
    # because the matrix payload never carries it.
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    test_state.state.app_settings.budget_mode = "reckless"
    clip = write_mp4(tmp_path / "clip.mp4", seconds=6)
    asset_id = _upload_video(client, clip)["id"]
    response = client.post(
        "/api/generations/edit/preflight",
        json={
            "feature": "extend",
            "assetId": asset_id,
            "startTime": 0,
            "mode": "end",
            "candidates": [{"key": "original", "width": None, "height": None}],
            "durations": [4],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["budgetMode"] == "reckless"


def test_iclora_preflight_echoes_live_budget_mode(
    test_state, client: TestClient, create_fake_model_files, tmp_path: Path
) -> None:
    # The bug is IC-LoRA pickers rendering disabled-vs-warned off a stale mode
    # because the preflight payload never carries it.
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=12, total_ram_gib=64)
    test_state.state.app_settings.budget_mode = "reckless"
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    frames = compute_num_frames(10, 24)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * frames, fps=24, width=960, height=540)
    )
    response = client.post(
        "/api/ic-lora/preflight",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "durations": [5, 20],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["budgetMode"] == "reckless"


def test_legacy_load_is_annotated_only_while_reckless(
    test_state, create_fake_model_files
) -> None:
    # The bug is a Linux job the picker marks unmeasured enqueueing with no
    # bet, so the queue never shows it and a later check can still block it.
    create_fake_model_files()
    test_state.config.system = "Linux"
    test_state.config.vram_gb = 24
    test_state.config.vram_bytes = 24 * _GIB
    test_state.config.total_ram_bytes = 64 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    handler = test_state.video_generation
    kwargs = {
        "offering": "ltx-2.5-fast",
        "resolution": "540p",
        "aspect_ratio": "16:9",
        "duration": 5,
        "fps": 24,
    }
    test_state.state.app_settings.budget_mode = "standard"
    assert handler.preflight_queued_fast_job(**kwargs) is None
    test_state.state.app_settings.budget_mode = "reckless"
    risk = handler.preflight_queued_fast_job(**kwargs)
    assert risk is not None and risk.tier == "unmeasured"


def test_flip_to_standard_still_streams_the_confirmed_bet(
    test_state, client: TestClient, create_fake_model_files
) -> None:
    # The bug is a mode flip re-blocking a job the user already confirmed.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.state.app_settings.budget_mode = "reckless"
    response = client.post(
        "/api/generations/text-to-video",
        json=_t2v_body(resolution="1080p", duration=5, aspect_ratio="21:9"),
    )
    assert response.status_code == 200, response.text
    record = SqliteStore(test_state.config.app_data_dir).get_generation(response.json()["id"])
    assert record is not None
    test_state.state.app_settings.budget_mode = "standard"
    handler = test_state.video_generation
    width, height = budget_shape(
        local_caps("ltx-2.5-22b-distilled"), "1080p", "21:9", mode="video"
    )
    frames = compute_num_frames(5, 24)
    job_filter = handler._local_job_filter(model_id="ltx-2.5-22b-distilled")
    mode, _risk = handler._video_job_load_mode(
        width,
        height,
        frames,
        job_filter=job_filter,
        allow_risk=spec_risk_allowance(record.spec),
    )
    assert mode == "streaming_models_loading"
    with pytest.raises(HTTPError) as exc_info:
        handler._video_job_load_mode(
            width,
            height,
            frames,
            job_filter=job_filter,
            allow_risk=False,
        )
    assert exc_info.value.status_code == 422


def test_unmeasured_note_does_not_waive_a_vram_shortfall(
    test_state, create_fake_model_files
) -> None:
    # The bug is an unmeasured note streaming a cell that does not fit.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16)
    test_state.state.app_settings.budget_mode = "standard"
    handler = test_state.video_generation
    width, height = budget_shape(
        local_caps("ltx-2.5-22b-distilled"), "1080p", "21:9", mode="video"
    )
    frames = compute_num_frames(5, 24)
    job_filter = handler._local_job_filter(model_id="ltx-2.5-22b-distilled")
    with pytest.raises(HTTPError) as exc_info:
        handler._video_job_load_mode(
            width,
            height,
            frames,
            job_filter=job_filter,
            allow_risk=spec_risk_allowance({"_risk": _stored_risk("unmeasured")}),
        )
    assert exc_info.value.status_code == 422
    mode, _risk = handler._video_job_load_mode(
        width,
        height,
        frames,
        job_filter=job_filter,
        allow_risk=spec_risk_allowance({"_risk": _stored_risk("over")}),
    )
    assert mode == "streaming_models_loading"


def test_unmeasured_note_still_runs_a_job_that_loads(
    test_state, create_fake_model_files
) -> None:
    # The bug is treating "unmeasured route" as "do not run."
    create_fake_model_files()
    test_state.config.system = "Linux"
    test_state.config.vram_gb = 24
    test_state.config.vram_bytes = 24 * _GIB
    test_state.config.total_ram_bytes = 64 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    test_state.state.app_settings.budget_mode = "standard"
    handler = test_state.video_generation
    width, height = budget_shape(
        local_caps("ltx-2.5-22b-distilled"), "540p", "16:9", mode="video"
    )
    frames = compute_num_frames(5, 24)
    job_filter = handler._local_job_filter(model_id="ltx-2.5-22b-distilled")
    mode, _risk = handler._video_job_load_mode(
        width,
        height,
        frames,
        job_filter=job_filter,
        allow_risk=spec_risk_allowance({"_risk": _stored_risk("unmeasured")}),
    )
    assert mode == "streaming_models_loading"


def test_iclora_preflight_names_a_downgradable_cell_in_standard(
    test_state, client: TestClient, create_fake_model_files, tmp_path: Path
) -> None:
    # The bug is a failing preflight cell looking identical for a hard refusal
    # and a cell Reckless would run. 32 GB RAM is under the IC-LoRA floor, so
    # the clip fails on memory and Reckless could still run it.
    create_fake_model_files()
    _configure_windows_edit(test_state, vram_gib=24, total_ram_gib=32)
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    frames = compute_num_frames(20, 24)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * frames, fps=24, width=1920, height=1080)
    )
    response = client.post(
        "/api/ic-lora/preflight",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "durations": [20],
        },
    )
    assert response.status_code == 200, response.text
    cells = response.json()["cells"]
    warned = [cell for cell in cells if cell["tier"] is not None]
    assert warned, cells
    assert all(cell["ok"] is False for cell in warned)
