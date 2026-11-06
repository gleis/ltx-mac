#!/usr/bin/env bash
# prepare-python.sh
# Downloads a standalone Python and installs all dependencies for macOS/Linux distribution.
#
# Dependencies are read from uv.lock (via `uv export`) — pyproject.toml is the
# single source of truth. No hardcoded dependency lists.
#
# Uses python-build-standalone (https://github.com/astral-sh/python-build-standalone)
# which provides relocatable Python builds for macOS and Linux.
#
# Prerequisites:
#   - uv must be installed (https://docs.astral.sh/uv/)
#   - curl must be available
#   - git must be available (for git-based Python packages)

set -euo pipefail

# ============================================================
# Configuration
# ============================================================
PYTHON_VERSION="${PYTHON_VERSION:-$(cat "$(dirname "$0")/../backend/.python-version" | tr -d '[:space:]')}"
PBS_TAG="${PBS_TAG:-20261026}"
OUTPUT_DIR="python-embed"
ARCH="${ARCH:-$(uname -m)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
OUTPUT_PATH="$PROJECT_DIR/$OUTPUT_DIR"
TEMP_DIR="$(mktemp -d)"

# Map architecture names for python-build-standalone
case "$ARCH" in
  arm64|aarch64) PBS_ARCH="aarch64" ;;
  x86_64|amd64)  PBS_ARCH="x86_64" ;;
  *) echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;;
esac

# Detect OS for python-build-standalone target triple
case "$(uname -s)" in
  Darwin) PBS_OS="apple-darwin" ;;
  Linux)  PBS_OS="unknown-linux-gnu" ;;
  *)      echo "ERROR: Unsupported OS: $(uname -s)"; exit 1 ;;
esac

PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/cpython-${PYTHON_VERSION}+${PBS_TAG}-${PBS_ARCH}-${PBS_OS}-install_only_stripped.tar.gz"

PLATFORM_LABEL="$(uname -s) ($ARCH)"

echo "========================================"
echo "  LTX Video - Python Environment Setup"
echo "  Platform: $PLATFORM_LABEL"
echo "  Python: $PYTHON_VERSION"
echo "========================================"

# ============================================================
# Step 1: Verify prerequisites
# ============================================================
echo ""
echo "Step 1: Verifying prerequisites..."

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it: https://docs.astral.sh/uv/"
    exit 1
fi
echo "  uv: $(command -v uv)"

if ! command -v curl &>/dev/null; then
    echo "ERROR: curl not found."
    exit 1
fi
echo "  curl: $(command -v curl)"

if ! command -v git &>/dev/null; then
    echo "ERROR: git not found (needed for git-based Python packages)."
    exit 1
fi
echo "  git: $(command -v git)"

# ============================================================
# Step 2: Generate requirements.txt from uv.lock
# ============================================================
echo ""
echo "Step 2: Generating requirements.txt from uv.lock..."

REQUIREMENTS_FILE="$BACKEND_DIR/requirements-dist.txt"

# Export pinned deps, excluding the project itself.
# Platform markers in pyproject.toml auto-exclude irrelevant deps
# (e.g. triton-windows on Linux/macOS, triton on macOS/Windows).
uv export --frozen --no-editable --no-emit-project \
    --no-header --no-annotate \
    --project "$BACKEND_DIR" \
    > "$REQUIREMENTS_FILE"

DEP_COUNT=$(grep -c '^\S' "$REQUIREMENTS_FILE" || true)
echo "  Exported $DEP_COUNT dependencies from uv.lock"

# ============================================================
# Step 3: Prepare directories
# ============================================================
echo ""
echo "Step 3: Preparing directories..."

if [ -d "$OUTPUT_PATH" ]; then
    echo "  Removing existing $OUTPUT_DIR directory..."
    rm -rf "$OUTPUT_PATH"
fi

mkdir -p "$OUTPUT_PATH"

# ============================================================
# Step 4: Download and extract standalone Python
# ============================================================
echo ""
echo "Step 4: Downloading Python $PYTHON_VERSION standalone ($PBS_ARCH)..."
echo "  URL: $PBS_URL"

PYTHON_TAR="$TEMP_DIR/python-standalone.tar.gz"
curl -L --fail --progress-bar -o "$PYTHON_TAR" "$PBS_URL"
echo "  Downloaded Python standalone package"

# python-build-standalone extracts to a `python/` directory
echo "  Extracting..."
tar -xzf "$PYTHON_TAR" -C "$TEMP_DIR"

# Move contents from python/ into our output path
mv "$TEMP_DIR/python/"* "$OUTPUT_PATH/"
echo "  Extracted to $OUTPUT_PATH"

# Verify the Python binary exists
PYTHON_EXE="$OUTPUT_PATH/bin/python3"
if [ ! -f "$PYTHON_EXE" ]; then
    echo "ERROR: Python binary not found at $PYTHON_EXE"
    exit 1
fi

echo "  Python binary: $PYTHON_EXE"
"$PYTHON_EXE" --version

# ============================================================
# Step 6: Install all dependencies from requirements.txt
# ============================================================
echo ""
echo "Step 6: Installing dependencies from requirements.txt..."
echo "  (This may take a while — PyTorch + ML libraries are large)"

# Install only the exported graph. uv checks the registry hashes and immutable
# Git revisions; do not add a second package index or resolve new dependencies.
uv pip install --python "$PYTHON_EXE" --no-deps -r "$REQUIREMENTS_FILE" --quiet

echo "  All dependencies installed"

# ============================================================
# Step 7: Clean up
# ============================================================
echo ""
# macOS/Apple Silicon: pre-compile mps-sdpa's zero-copy `mpsgraph_zc` attention
# extension so it ships prebuilt (end-user Macs have no compiler). MUST run before
# the setuptools strip below — torch.utils.cpp_extension imports setuptools at load.
# Output goes INSIDE python-embed (mps-ext-prebuilt/) so it rides the CI python-embed
# cache and gets bundled with it; the runtime env LTX_MPS_EXT_PREBUILT_DIR points here.
if [ "$PBS_OS" = "apple-darwin" ] && [ "$PBS_ARCH" = "aarch64" ]; then
  echo "Step 6.5: Pre-building mps-sdpa mpsgraph_zc attention extension..."
  # Fatal, not a warning: without this prebuilt .so, end-user Macs (no compiler) fall
  # back to the leaking pyobjc attention backend and OOM on long generations.
  PYTHON="$PYTHON_EXE" bash "$SCRIPT_DIR/prebuild-mps-sdpa-ext.sh" "$OUTPUT_PATH/mps-ext-prebuilt"
fi

echo ""
echo "Step 7: Cleaning up..."

# Remove __pycache__ and .pyc files
find "$OUTPUT_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.pyc" -delete 2>/dev/null || true

# Remove pip cache and pip itself (not needed at runtime)
rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/pip 2>/dev/null || true
rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/pip-*.dist-info 2>/dev/null || true
# Keep setuptools on macOS: torch.utils.cpp_extension imports it at module top,
# and it's imported at runtime to load the prebuilt mpsgraph_zc attention extension.
if [ "$PBS_OS" != "apple-darwin" ]; then
  rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/setuptools 2>/dev/null || true
  rm -rf "$OUTPUT_PATH/lib/python"*/site-packages/setuptools-*.dist-info 2>/dev/null || true
fi

# Remove test directories to save space
find "$OUTPUT_PATH/lib" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find "$OUTPUT_PATH/lib" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Remove files only needed for building native extensions, not at runtime.
# This cuts ~14k files and speeds up macOS codesigning dramatically.
# NOTE: Linux needs .h files for sageattention/triton JIT compilation.
if [ "$PBS_OS" = "apple-darwin" ]; then
  rm -rf "$OUTPUT_PATH/include" "$OUTPUT_PATH/share" 2>/dev/null || true
  find "$OUTPUT_PATH/lib" -type d -name "include" -exec rm -rf {} + 2>/dev/null || true
  find "$OUTPUT_PATH" -name "*.h" -delete 2>/dev/null || true
  find "$OUTPUT_PATH" -name "*.cuh" -delete 2>/dev/null || true
  find "$OUTPUT_PATH" -name "*.cu" -delete 2>/dev/null || true
else
  # Linux: keep .h/.cuh/.cu files for triton/sageattention JIT, but remove other build artifacts
  rm -rf "$OUTPUT_PATH/share" 2>/dev/null || true
fi
find "$OUTPUT_PATH" -name "*.pyi" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.pxd" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.pyx" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.hpp" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.cpp" -delete 2>/dev/null || true
find "$OUTPUT_PATH" -name "*.cmake" -delete 2>/dev/null || true

# Remove temp directory and generated requirements file
rm -rf "$TEMP_DIR"
rm -f "$REQUIREMENTS_FILE"

echo "  Cleanup complete"

# ============================================================
# Step 8: Verify installation
# ============================================================
echo ""
echo "Step 8: Verifying installation..."

"$PYTHON_EXE" -c "
import sys
import platform
print(f'  Python: {sys.version}')
try:
    import torch
    print(f'  PyTorch: {torch.__version__}')
    if platform.system() == 'Darwin':
        mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        print(f'  MPS available: {mps}')
    elif platform.system() == 'Linux':
        cuda = torch.cuda.is_available()
        print(f'  CUDA available: {cuda}')
        if cuda:
            print(f'  CUDA version: {torch.version.cuda}')
except ImportError as e:
    print(f'  PyTorch import FAILED: {e}')
    sys.exit(1)
try:
    import fastapi
    print(f'  FastAPI: {fastapi.__version__}')
except ImportError as e:
    print(f'  FastAPI import FAILED: {e}')
    sys.exit(1)
try:
    import diffusers
    print(f'  Diffusers: {diffusers.__version__}')
except ImportError as e:
    print(f'  Diffusers import FAILED: {e}')
    sys.exit(1)
try:
    from ltx_pipelines import distilled
    print(f'  ltx-pipelines: OK')
except ImportError as e:
    print(f'  ltx-pipelines: FAILED - {e}')
    sys.exit(1)
"

# Calculate size
SIZE_BYTES=$(du -sb "$OUTPUT_PATH" 2>/dev/null | cut -f1 || du -sk "$OUTPUT_PATH" | awk '{print $1 * 1024}')
SIZE_GB=$(awk "BEGIN {printf \"%.2f\", $SIZE_BYTES / 1073741824}")

echo ""
echo "========================================"
echo "  Python environment ready!"
echo "  Location: $OUTPUT_PATH"
echo "  Size: ${SIZE_GB} GB"
echo "========================================"
