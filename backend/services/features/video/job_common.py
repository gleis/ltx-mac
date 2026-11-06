"""Shared queued-video-job helpers (t2v / i2v / a2v / retake / extend)."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeGuard, TypeVar

from pydantic import BaseModel, ValidationError

from api_types import (
    GenerationRisk,
    IcLoraRecipeResolution,
    IcLoraSynthesizedStoredParams,
    LTXLocalModelId,
    LTXVideoGenAspectRatio,
    OfferingId,
)
from runtime_config.ic_lora_high_res import high_res_cell
from _routes._errors import HTTPError
from runtime_config.ltx_capabilities import LtxCapabilityFeature, budget_pixels, local_caps, supports
from runtime_config.offerings import resolve_offering_local_model_id
from services.records import AssetRecord, CapabilityFailedError
from state.app_settings import MAX_GENERATION_SEED

INPUT_ASSET_NOT_FOUND = "INPUT_ASSET_NOT_FOUND"

TModel = TypeVar("TModel", bound=BaseModel)


class AssetLookup(Protocol):
    def get_asset(self, asset_id: str) -> AssetRecord | None: ...


def is_existing_file(path: str) -> bool:
    try:
        return Path(path).is_file()
    except OSError:
        return False


def looks_like_filesystem_path(detail: str) -> bool:
    stripped = detail.strip()
    if not stripped:
        return False
    if Path(stripped).is_absolute():
        return True
    return "/" in stripped or "\\" in stripped


def capability_failed_from_http(
    exc: HTTPError, *, unavailable: str = "input media is no longer available"
) -> CapabilityFailedError:
    if looks_like_filesystem_path(exc.detail):
        return CapabilityFailedError(unavailable, code=INPUT_ASSET_NOT_FOUND)
    return CapabilityFailedError(exc.detail, code=exc.code)


def model_from_spec(
    spec: Mapping[str, object], key: str, model: type[TModel]
) -> TModel:
    value = spec.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"generation spec requires a {key} object")
    return model.model_validate(value)


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def stored_generation_risk(spec: Mapping[str, object]) -> GenerationRisk | None:
    """The stored bet, or None when it is missing or junk.

    Junk fails closed: execute must not waive a limit off a corrupt row.
    """
    risk = spec.get("_risk")
    if not _is_str_object_dict(risk):
        return None
    try:
        return GenerationRisk.model_validate(risk)
    except ValidationError:
        return None


def spec_risk_allowance(spec: Mapping[str, object]) -> bool:
    """Whether the stored bet waives a later memory shortfall.

    Only ``over`` does. ``unmeasured`` means the route had no fitted curve,
    not that a VRAM or RAM miss may run. A missing field or junk fails closed.
    Executors pass this — never the live mode — so turning Reckless off does
    not cancel an over-budget job already in the queue.
    """
    risk = stored_generation_risk(spec)
    return risk is not None and risk.tier == "over"


def spec_unmeasured_bet(spec: Mapping[str, object]) -> bool:
    """Whether the stored bet is an unmeasured route, not a memory shortfall."""
    risk = stored_generation_risk(spec)
    return risk is not None and risk.tier == "unmeasured"


def resolve_generation_seed(seed: int | None) -> int:
    return seed if seed is not None else random.randint(0, MAX_GENERATION_SEED)


def resolve_job_local_model_id(
    models_dir: Path, offering: OfferingId
) -> LTXLocalModelId:
    resolved = resolve_offering_local_model_id(models_dir, offering)
    if resolved is None:
        raise HTTPError(409, "LTX_MODEL_NOT_INSTALLED")
    return resolved


def require_capable_local_model(
    models_dir: Path,
    offering: OfferingId,
    capability: LtxCapabilityFeature,
    *,
    unsupported_code: str,
    unsupported_message: str,
) -> LTXLocalModelId:
    local_id = resolve_job_local_model_id(models_dir, offering)
    if not supports(local_caps(local_id), capability):
        raise CapabilityFailedError(unsupported_message, code=unsupported_code)
    return local_id


def require_existing_asset(
    assets: AssetLookup,
    asset_id: str,
    *,
    media_kind: str,
    unavailable: str,
) -> AssetRecord:
    asset = assets.get_asset(asset_id)
    if asset is None or asset.media_kind != media_kind or not is_existing_file(asset.path):
        raise CapabilityFailedError(unavailable, code=INPUT_ASSET_NOT_FOUND)
    return asset


# The name of a recipe whose display name the caller cannot reach. The catalog owns the real name.
UNNAMED_RECIPE = "This recipe"


def high_res_unsupported_error(name: str) -> HTTPError:
    """The refusal of a recipe that cannot run as a tiled high cell. One wording for every recipe."""
    return HTTPError(
        422,
        f"{name} cannot render above 1080p. Pick a lower resolution.",
        code="IC_LORA_HIGH_RES_UNSUPPORTED",
    )


def denoise_canvas_for(
    resolution: IcLoraRecipeResolution,
    aspect_ratio: LTXVideoGenAspectRatio,
    *,
    name: str,
    local_model_id: LTXLocalModelId = "ltx-2.5-22b-distilled",
) -> tuple[int, int]:
    """The size the model denoises. The sheet and the control clip use it."""
    # A high cell runs as spatial tiles, and each tile sees only its own slice of the
    # sheet, so the panels would not stay whole. Queued or retried specs from before
    # the cells were removed reach this helper, so it owns the rejection.
    if high_res_cell(resolution) is not None:
        raise high_res_unsupported_error(name)
    try:
        return budget_pixels(
            local_caps(local_model_id), resolution, aspect_ratio, mode="video"
        )
    except KeyError as exc:
        raise HTTPError(400, "INVALID_LOCAL_RESOLUTION", code="INVALID_LOCAL_RESOLUTION") from exc


def denoise_canvas(
    local_model_id: LTXLocalModelId, params: IcLoraSynthesizedStoredParams, *, name: str
) -> tuple[int, int]:
    return denoise_canvas_for(
        params.resolution, params.aspectRatio, name=name, local_model_id=local_model_id
    )
