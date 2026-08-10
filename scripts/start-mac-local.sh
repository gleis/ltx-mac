#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

export LTX_APP_DATA_DIR="${LTX_APP_DATA_DIR:-$REPO_ROOT/.ltx-data}"
export LTX_MLX_PATH="${LTX_MLX_PATH:-$REPO_ROOT/.ltx-data/ltx-2-mlx}"
export PNPM_STORE_DIR="${PNPM_STORE_DIR:-$REPO_ROOT/.pnpm-store}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.ltx-data/cache}"

mkdir -p "$LTX_APP_DATA_DIR" "$PNPM_STORE_DIR" "$XDG_CACHE_HOME"

if [[ ! -x "$REPO_ROOT/node_modules/.bin/vite" ]]; then
  echo "LTX Desktop dependencies are not installed."
  echo "Run this from the project directory first: pnpm install"
  exit 1
fi

if [[ ! -d "$LTX_MLX_PATH" ]]; then
  echo "Local MLX runtime was not found at: $LTX_MLX_PATH"
  echo "Run this from the project directory first: scripts/setup-mac-mlx.sh"
  exit 1
fi

echo "Starting LTX Desktop Local MLX"
echo "Project: $REPO_ROOT"
echo "App data: $LTX_APP_DATA_DIR"
echo "MLX runtime: $LTX_MLX_PATH"
echo

exec "$REPO_ROOT/node_modules/.bin/vite" "$@"
