"""Memory-safe decoder settings shared by setup validation and the MLX helper."""
from __future__ import annotations
from collections.abc import MutableMapping


def apply_decode_policy(env: MutableMapping[str, str], num_frames: int, user_override: str | None) -> None:
    if user_override is not None:
        return
    threshold = int(env.get("LTX_VAE_STREAMING_AUTO_MAX_FRAMES", "0"))
    env["LTX_VAE_STREAMING"] = "0" if num_frames <= threshold else "1"
    env.setdefault("LTX_VAE_TILE_FRAMES", "32")
    env.setdefault("LTX_VAE_TILE_OVERLAP_FRAMES", "24")
