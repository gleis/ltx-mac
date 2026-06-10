#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DATA_DIR="${LTX_APP_DATA_DIR:-"$ROOT/.ltx-data"}"
MLX_ROOT="${LTX_MLX_PATH:-"$APP_DATA_DIR/ltx-2-mlx"}"
MODELS_DIR="${LTX_MODELS_DIR:-"$APP_DATA_DIR/models"}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-"$APP_DATA_DIR/uv-cache"}"
export HF_HOME="${HF_HOME:-"$APP_DATA_DIR/hf-home"}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-"$APP_DATA_DIR/cache"}"
UV_BIN="${UV_BIN:-}"

if [[ -z "$UV_BIN" ]]; then
  if [[ -x "$ROOT/.venv-tools/bin/uv" ]]; then
    UV_BIN="$ROOT/.venv-tools/bin/uv"
  else
    UV_BIN="uv"
  fi
fi

echo "== LTX Desktop Mac-local MLX setup =="
echo "app data: $APP_DATA_DIR"
echo "mlx repo: $MLX_ROOT"
echo "models:   $MODELS_DIR"
echo "cache:    $APP_DATA_DIR/{uv-cache,hf-home,cache}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This setup is for Apple Silicon Macs only." >&2
  exit 1
fi

mkdir -p "$APP_DATA_DIR" "$MODELS_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"

if [[ ! -d "$MLX_ROOT/.git" ]]; then
  git clone https://github.com/dgrauet/ltx-2-mlx.git "$MLX_ROOT"
fi

cd "$MLX_ROOT"
git fetch --tags origin
git checkout v0.14.0

if [[ ! -x env/bin/python3.11 ]]; then
  rm -rf env
  "$UV_BIN" venv --python 3.11 --seed env
fi

"$UV_BIN" pip install --python env/bin/python \
  'mlx==0.31.1' 'mlx-lm==0.31.1' 'mlx-metal==0.31.1'
"$UV_BIN" pip install --python env/bin/python \
  --reinstall ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx ./packages/ltx-trainer
"$UV_BIN" pip install --python env/bin/python \
  pyyaml pydantic tqdm rich pillow numpy 'huggingface-hub>=1.5.0,<2.0' 'hf_transfer>=0.1.6'

PATCH_SCRIPT="$ROOT/backend/services/fast_video_pipeline/patch_ltx_codec.py"
if [[ -f "$PATCH_SCRIPT" ]]; then
  if ! env/bin/python3.11 "$PATCH_SCRIPT"; then
    echo "Warning: optional MLX codec/memory patch failed; continuing with the stock ltx-2-mlx runtime." >&2
  fi
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

env/bin/hf download dgrauet/ltx-2.3-mlx-q4 \
  --local-dir "$MODELS_DIR/ltx-2.3-mlx-q4" \
  --include '*.json' \
  --include transformer-distilled.safetensors \
  --include connector.safetensors \
  --include vae_decoder.safetensors \
  --include vae_encoder.safetensors \
  --include audio_vae.safetensors \
  --include vocoder.safetensors

env/bin/hf download mlx-community/gemma-3-12b-it-4bit \
  --local-dir "$MODELS_DIR/gemma-3-12b-it-4bit"

echo
echo "Done. For dev runs, use:"
echo "  export LTX_APP_DATA_DIR=\"$APP_DATA_DIR\""
echo "  export LTX_MLX_PATH=\"$MLX_ROOT\""
echo "  pnpm dev"
