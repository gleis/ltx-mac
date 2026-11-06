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
# MLX 0.32.4 publishes its Apple Silicon wheels for macOS 26 or later.
macos_major="$(sw_vers -productVersion | cut -d. -f1)"
if (( macos_major < 26 )); then
  echo "The pinned MLX runtime requires macOS 26 or later. Update macOS before setup." >&2
  exit 1
fi

mkdir -p "$APP_DATA_DIR" "$MODELS_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"

if [[ ! -d "$MLX_ROOT/.git" ]]; then
  git clone https://github.com/dgrauet/ltx-2-mlx.git "$MLX_ROOT"
fi

cd "$MLX_ROOT"
git fetch --tags origin
# Immutable source revision; v0.14 API compatibility is maintained until the
# v0.18 pipeline migration and real image-to-video validation are complete.
git checkout --detach b35254a9288e838afc574e8002432815a04f16f4

# Use the same interpreter version as the backend and hashed third-party graph.
PYTHON_VERSION="$(tr -d '[:space:]' < "$ROOT/backend/.python-version")"
"$UV_BIN" venv --python "$PYTHON_VERSION" --allow-existing env
"$UV_BIN" pip sync --python env/bin/python --require-hashes "$ROOT/requirements/mlx.lock"
# Hatchling requires readmes inside each package. Keep the pinned source code
# unchanged and copy only its shared readme into the package build roots.
env/bin/python - <<'PYBUILD'
from pathlib import Path
root = Path.cwd()
for name in ("ltx-core-mlx", "ltx-pipelines-mlx", "ltx-trainer"):
    project = root / "packages" / name / "pyproject.toml"
    project.write_text(project.read_text().replace('readme = "../../README.md"', 'readme = "README.md"'))
    (project.parent / "README.md").write_bytes((root / "README.md").read_bytes())
PYBUILD
"$UV_BIN" pip install --python env/bin/python --no-deps --no-build-isolation \
  ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx ./packages/ltx-trainer
"$UV_BIN" pip check --python env/bin/python
env/bin/python "$ROOT/backend/services/fast_video_pipeline/prepare_mlx_runtime.py"



export HF_HUB_ENABLE_HF_TRANSFER=1

env/bin/hf download dgrauet/ltx-2.3-mlx-q4 \
  --local-dir "$MODELS_DIR/ltx-2.3-mlx-q4" \
  --include '*.json' \
  --include transformer-distilled.safetensors \
  --include transformer-dev.safetensors \
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
