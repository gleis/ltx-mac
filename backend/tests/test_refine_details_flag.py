"""Refine Details needs the highResVideoToVideo flag: every path refuses or hides it while the flag is off."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from starlette.testclient import TestClient

from api_types import IcLoraCatalogItem, parse_lora_catalog
from handlers.base import resolve_models_dir
from runtime_config.model_download_specs import resolve_ic_lora_path
from tests.http_error_assertions import assert_http_error

_REFINE = "refine-details"
_ALPHA = "alpha-gen"
_CATALOG = Path(__file__).resolve().parents[1] / "runtime_config" / "lora_catalog.json"


def _mp4(path: Path) -> Path:
    writer = imageio.get_writer(str(path), fps=8, codec="libx264", macro_block_size=None)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for _ in range(8):
        writer.append_data(frame)
    writer.close()
    return path


def _set_flag(client: TestClient, enabled: bool) -> None:
    response = client.request("PATCH", "/api/feature-flags", json={"highResVideoToVideo": enabled})
    assert response.status_code == 200, response.text


def _install(fake_services, test_state, ic_lora_id: str, *, weights: bool = True) -> IcLoraCatalogItem:
    """Add the shipped catalog entry to the fake catalog and write its weights."""
    item = next(
        entry
        for entry in parse_lora_catalog(_CATALOG.read_text()).ic_loras
        if entry.id == ic_lora_id
    )
    fake_services.lora_catalog_provider._catalog.ic_loras.append(item)
    if weights:
        models_dir = resolve_models_dir(test_state.state, test_state.config)
        path = resolve_ic_lora_path(models_dir, item.id, item.download.variants[0].filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
    return item


def _create_recipe(client: TestClient, recipe_id: str, video_id: str, resolution: str = "720p"):
    return client.post(
        f"/api/generations/ic-lora-recipes/{recipe_id}",
        json={
            "params": {"prompt": "restore fine detail", "model": "ltx-2.5-fast", "resolution": resolution},
            "inputs": {"video": {"assetId": video_id}},
        },
    )


def _listed_ids(client: TestClient) -> list[str]:
    response = client.get("/api/ic-loras")
    assert response.status_code == 200, response.text
    return [row["ic_lora"]["id"] for row in response.json()["ic_loras"]]


def test_catalog_list_hides_refine_details_until_the_flag_is_on(
    client: TestClient, fake_services, test_state, create_fake_model_files
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE, weights=False)

    assert _REFINE not in _listed_ids(client)

    _set_flag(client, True)
    assert _REFINE in _listed_ids(client)


def test_download_of_refine_details_is_unknown_until_the_flag_is_on(
    client: TestClient, fake_services, test_state, create_fake_model_files
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE, weights=False)

    refused = client.post("/api/ic-loras/download", json={"ic_lora_id": _REFINE})
    assert refused.status_code == 404

    _set_flag(client, True)
    accepted = client.post("/api/ic-loras/download", json={"ic_lora_id": _REFINE})
    assert accepted.status_code == 200, accepted.text


def test_generate_and_preflight_refuse_refine_details_until_the_flag_is_on(
    client: TestClient, fake_services, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    clip = _mp4(tmp_path / "clip.mp4")
    generate = {
        "ic_lora_id": _REFINE,
        "input_path": str(clip),
        "prompt": "",
        "conditioning_type": "custom",
    }
    preflight = {"ic_lora_id": _REFINE, "input_path": str(clip), "durations": [1.0]}

    for path, body in (("/api/ic-lora/generate", generate), ("/api/ic-lora/preflight", preflight)):
        assert_http_error(
            client.post(path, json=body),
            status_code=422,
            code="HIGH_RES_NOT_ENABLED",
            message="HIGH_RES_NOT_ENABLED",
        )

    _set_flag(client, True)
    allowed = client.post("/api/ic-lora/preflight", json=preflight)
    assert allowed.status_code == 200, allowed.text


def test_home_create_refuses_refine_details_until_the_flag_is_on(
    client: TestClient, fake_services, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()

    assert_http_error(
        _create_recipe(client, _REFINE, video["id"]),
        status_code=422,
        code="HIGH_RES_NOT_ENABLED",
        message="HIGH_RES_NOT_ENABLED",
    )

    _set_flag(client, True)
    accepted = _create_recipe(client, _REFINE, video["id"])
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["spec"]["params"]["loras"][0]["catalogId"] == _REFINE


def test_recent_features_skip_refine_details_before_the_limit_while_the_flag_is_off(
    client: TestClient, fake_services, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _ALPHA)
    _install(fake_services, test_state, _REFINE)
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
    assert _create_recipe(client, _ALPHA, video["id"], "540p").status_code == 200
    _set_flag(client, True)
    assert _create_recipe(client, _REFINE, video["id"]).status_code == 200

    shown = client.get("/api/generations/recent-features", params={"limit": 1})
    assert shown.json() == [_REFINE]

    _set_flag(client, False)
    hidden = client.get("/api/generations/recent-features", params={"limit": 1})
    assert hidden.json() == [_ALPHA]
