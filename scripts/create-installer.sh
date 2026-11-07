#!/usr/bin/env bash
# create-installer.sh
# Runs electron-builder to produce the installer (dmg/exe).
# This is the ONLY build stage that needs code-signing secrets.
#
# Expects the frontend to be built and python-embed to be ready.
# See local-build.sh for the convenience wrapper that runs all stages.
#
# Usage:
#   bash scripts/create-installer.sh [options]
#
# Options:
#   --platform mac|win   Target platform (auto-detected if omitted)
#   --publish <mode>     Publish mode for electron-builder (always|never|onTag)
#   --unpack             Build unpacked app only (faster, no installer/dmg)

set -euo pipefail

# ============================================================
# Parse arguments
# ============================================================
UNPACK=false
PLATFORM=""
PUBLISH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unpack) UNPACK=true ;;
    --publish)
      PUBLISH="$2"
      shift
      ;;
    --platform)
      PLATFORM="$2"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--platform mac|win] [--publish always|never|onTag] [--unpack]"
      exit 1
      ;;
  esac
  shift
done

# Auto-detect platform if not specified
if [ -z "$PLATFORM" ]; then
  case "$(uname -s)" in
    Darwin)          PLATFORM="mac" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="win" ;;
    Linux)           PLATFORM="linux" ;;
    *)               echo "ERROR: Could not detect platform. Use --platform mac|win"; exit 1 ;;
  esac
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RELEASE_DIR="$PROJECT_DIR/release"

cd "$PROJECT_DIR"

# The installer config ships this dependency-graph identifier. Generate it for
# local builds too; it is not an archive-integrity checksum.
node --input-type=module - <<'NODE'
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
const hash = createHash('sha256')
for (const file of ['backend/.python-version', 'backend/pyproject.toml', 'backend/uv.lock']) {
  hash.update(file).update('\0').update(readFileSync(file)).update('\0')
}
writeFileSync('python-deps-hash.txt', hash.digest('hex') + '\n')
NODE

# ============================================================
# Verify prerequisites
# ============================================================
if [ ! -d "dist" ] || [ ! -d "dist-electron" ] || [ ! -f "dist-remote/index.html" ]; then
  echo "ERROR: Frontend not built. Run local-build.sh or 'pnpm run build:frontend' first."
  exit 1
fi

if [ "$PLATFORM" != "linux" ] && [ ! -d "python-embed" ]; then
  echo "ERROR: Python environment not found. Run local-build.sh or prepare-python.sh first."
  exit 1
fi

# ============================================================
# Build with electron-builder
# ============================================================
BUILDER_ARGS=""
if [ "$PLATFORM" = mac ] && [ ! -x "$PROJECT_DIR/mlx-runtime/python/bin/python3" ]; then
  echo "ERROR: MLX runtime missing. Run scripts/prepare-mlx-bundle.sh first." >&2
  exit 1
fi
case "$PLATFORM" in
  mac)   BUILDER_ARGS="--mac" ;;
  win)   BUILDER_ARGS="--win" ;;
  linux) BUILDER_ARGS="--linux" ;;
esac

if [ "$UNPACK" = true ]; then
  echo "Packaging unpacked app (fast mode)..."
  pnpm exec electron-builder $BUILDER_ARGS --dir
else
  PUBLISH_ARGS=""
  if [ -n "$PUBLISH" ]; then
    PUBLISH_ARGS="--publish $PUBLISH"
  fi
  echo "Packaging installer..."
  pnpm exec electron-builder $BUILDER_ARGS $PUBLISH_ARGS
fi
echo ""

# ============================================================
# Summary
# ============================================================
echo "========================================"
echo "  Build Complete!"
echo "========================================"

if [ "$UNPACK" = true ]; then
  case "$PLATFORM" in
    mac)
      echo ""
      echo "Unpacked app ready!"
      echo "Run: open \"$RELEASE_DIR/mac-arm64/LTX Desktop.app\""
      ;;
    win)
      echo ""
      echo "Unpacked app ready!"
      echo "Run: $RELEASE_DIR/win-unpacked/LTX Desktop.exe"
      ;;
    linux)
      echo ""
      echo "Unpacked app ready!"
      LINUX_UNPACKED="$RELEASE_DIR/linux-unpacked"
      [ -d "$RELEASE_DIR/linux-arm64-unpacked" ] && LINUX_UNPACKED="$RELEASE_DIR/linux-arm64-unpacked"
      echo "Run: $LINUX_UNPACKED/ltx-desktop"
      ;;
  esac
else
  echo ""
  echo "Output: $RELEASE_DIR/"
  ls -1 "$RELEASE_DIR/" 2>/dev/null | head -10
fi

echo ""
echo "Note: AI models (~150GB) will be downloaded on first run."
