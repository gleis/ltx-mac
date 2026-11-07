#!/usr/bin/env bash
# Build a relocatable MLX runtime; model weights are downloaded by the app.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || exit 0
if (( $(sw_vers -productVersion | cut -d. -f1) < 26 )); then
  echo 'The pinned MLX bundle requires macOS 26 or later.' >&2
  exit 1
fi
UV_BIN=${UV_BIN:-uv}
PYTHON_VERSION=$(tr -d '[:space:]' < "$ROOT/backend/.python-version")
PBS_TAG=${PBS_TAG:-20261026}
OUT="$ROOT/mlx-runtime"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$OUT"
curl -fLsS "https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/cpython-$PYTHON_VERSION+$PBS_TAG-aarch64-apple-darwin-install_only_stripped.tar.gz" -o "$STAGE/python.tar.gz"
tar -xzf "$STAGE/python.tar.gz" -C "$STAGE"
# A separate standalone prefix prevents backend/MLX dependency version conflicts.
rm -rf "$OUT/python"
mv "$STAGE/python" "$OUT/python"
curl -fLsS https://codeload.github.com/dgrauet/ltx-2-mlx/tar.gz/b35254a9288e838afc574e8002432815a04f16f4 -o "$STAGE/source.tar.gz"
echo "f86a2385292b4b52f425858dbf6192b3422c72794afa4ece5b6a549843ec5f41  $STAGE/source.tar.gz" | shasum -a 256 -c -
mkdir "$STAGE/source"
tar -xzf "$STAGE/source.tar.gz" --strip-components=1 -C "$STAGE/source"
PY="$OUT/python/bin/python3"
"$UV_BIN" pip install --python "$PY" --require-hashes -r "$ROOT/requirements/mlx.lock"
"$PY" - "$STAGE/source" <<'PYBUILD'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for name in ('ltx-core-mlx', 'ltx-pipelines-mlx', 'ltx-trainer'):
    project = root / 'packages' / name / 'pyproject.toml'
    project.write_text(project.read_text().replace('readme = "../../README.md"', 'readme = "README.md"'))
    (project.parent / 'README.md').write_bytes((root / 'README.md').read_bytes())
PYBUILD
"$UV_BIN" pip install --python "$PY" --no-deps --no-build-isolation \
  "$STAGE/source/packages/ltx-core-mlx" "$STAGE/source/packages/ltx-pipelines-mlx" "$STAGE/source/packages/ltx-trainer"
"$UV_BIN" pip check --python "$PY"
"$PY" "$ROOT/backend/services/fast_video_pipeline/prepare_mlx_runtime.py"
cp "$STAGE/source/pyproject.toml" "$OUT/pyproject.toml"
cp "$STAGE/source/LICENSE" "$OUT/LICENSE"
printf '%s\n' b35254a9288e838afc574e8002432815a04f16f4 > "$OUT/SOURCE_REVISION"
find "$OUT" -type d -name __pycache__ -prune -exec rm -rf {} +
echo "Prepared relocatable MLX runtime: $OUT"
