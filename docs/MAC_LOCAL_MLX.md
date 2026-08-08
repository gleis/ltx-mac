# Mac Local MLX Runbook

This document describes how to run this fork of LTX Desktop locally on Apple
Silicon without the LTX video-generation API. It covers setup, storage,
runtime behavior, settings, and troubleshooting.

## Target Hardware

The first milestone is aimed at:

- Apple Silicon Mac.
- macOS 13 or newer.
- About 36 GB unified memory.
- External/project drive with enough free space for models, runtimes, caches,
  and generated outputs.

The runtime is tuned for the compact MLX Q4 path, not the full CUDA/PyTorch
model path.

## What Runs Locally

The Mac-local path uses:

- `dgrauet/ltx-2.3-mlx-q4` for the compact LTX model.
- `mlx-community/gemma-3-12b-it-4bit` for local text encoding and local prompt
  enhancement.
- `dgrauet/ltx-2-mlx` checked out at `v0.14.0`.
- A long-lived helper process launched from
  `backend/services/fast_video_pipeline/mlx_fast_video_pipeline.py`.

The generation order is:

1. Optional prompt enhancement through the local Gemma helper.
2. Local prompt/text encoding.
3. Local MLX rendering.
4. Output handoff back to the backend/app.

## Storage Layout

Development builds default to project-local app data:

```text
<repo>/.ltx-data/
```

The setup script creates this layout:

```text
.ltx-data/
  ltx-2-mlx/
    env/                         # Python 3.11 MLX runtime env
  models/
    ltx-2.3-mlx-q4/
    gemma-3-12b-it-4bit/
  uv-cache/
  hf-home/
  cache/
  outputs/
```

The folder is ignored by Git. Do not commit it.

To force the app and setup script to use a different location:

```bash
export LTX_APP_DATA_DIR="/Volumes/X10Pro4T/projects/ltx-mac/.ltx-data"
```

You may also override specific pieces:

```bash
export LTX_MLX_PATH="/Volumes/X10Pro4T/projects/ltx-mac/.ltx-data/ltx-2-mlx"
export LTX_MODELS_DIR="/Volumes/X10Pro4T/projects/ltx-mac/.ltx-data/models"
export UV_CACHE_DIR="/Volumes/X10Pro4T/projects/ltx-mac/.ltx-data/uv-cache"
export HF_HOME="/Volumes/X10Pro4T/projects/ltx-mac/.ltx-data/hf-home"
export XDG_CACHE_HOME="/Volumes/X10Pro4T/projects/ltx-mac/.ltx-data/cache"
```

## One-Time Setup

From the repository root:

```bash
corepack enable
pnpm install
scripts/setup-mac-mlx.sh
```

The script:

1. Creates `.ltx-data/`.
2. Clones `dgrauet/ltx-2-mlx`.
3. Checks out `v0.14.0`.
4. Creates the MLX Python 3.11 environment.
5. Installs pinned MLX packages.
6. Applies the local codec/memory patch when available.
7. Downloads the MLX Q4 LTX model.
8. Downloads the 4-bit Gemma model.

If `uv` is installed inside the repo helper environment, the script uses
`.venv-tools/bin/uv`; otherwise it falls back to `uv` on `PATH`.

## Running

Start the app in development with project-local storage and the prepared MLX
runtime checkout:

```bash
export LTX_APP_DATA_DIR="$PWD/.ltx-data"
export LTX_MLX_PATH="$PWD/.ltx-data/ltx-2-mlx"
pnpm dev
```

If `pnpm dev` attempts to rerun dependency installation and aborts with
`ERR_PNPM_ABORTED_REMOVE_MODULES_DIR_NO_TTY`, use the installed Vite binary
directly:

```bash
export LTX_APP_DATA_DIR="$PWD/.ltx-data"
export LTX_MLX_PATH="$PWD/.ltx-data/ltx-2-mlx"
./node_modules/.bin/vite
```

The backend should select `mac_mlx_q4` automatically on Apple Silicon with
enough unified memory.

## In-App Settings

Open Settings after launch.

Recommended local Mac settings:

- Generation mode: local, when the app offers it.
- Prompt Enhancer: enabled if you want the local Gemma helper to rewrite prompts
  before local encoding/rendering.
- Mac MLX Speed:
  - `Quality`: default, exact sampler.
  - `Boost`: faster text-to-video, skips stable middle denoise steps.
  - `Turbo`: fastest/aggressive text-to-video, more likely to show artifacts.

Start with `Quality` for comparisons. Try `Boost` for ordinary iteration once
you have a text-to-video prompt that works. Treat `Turbo` as experimental.
Image-to-video always uses `Quality` in this fork because accelerated I2V can
produce tiled conditioning artifacts.

## Recommended Generation Settings

| Resolution | Duration | Notes |
| --- | --- | --- |
| 540p | 5-20 seconds | Good for longer experiments and prompt iteration |
| 720p | 5-20 seconds | Best practical default; 10-second output is confirmed usable on the 36 GB target |
| 1080p | 5 seconds | Current conservative quality gate for local Mac output |

A 720p 10-second text-to-video render has been confirmed to complete and produce
a decent usable result on the target 36 GB Mac. 1080p 10-second generation has
also completed, but extended quality still needs more validation before treating
it as a default recommendation.

## Prompt Enhancement

The prompt helper is in:

```text
Settings -> Prompt Enhancer
```

For this fork:

- API text encoding can use LTX API prompt enhancement when an API key is
  configured.
- Local Mac MLX generation uses the downloaded Gemma helper and does not require
  an LTX API key for prompt rewriting.
- The local flow needs a rewritten prompt string before local encoding, so it
  uses the helper's `enhance_prompt` action rather than the LTX API
  `/v1/prompt-embedding` endpoint.

## Scene Queue

Gen Space includes a local scene queue for unattended batch rendering. In video
mode, expand `Scene Queue` above the prompt bar, paste one prompt per line, click
`Add`, then click the play button. The queue runs scenes sequentially so the Mac
only renders one clip at a time. Pause stops after the current clip completes.

Queue items keep a snapshot of the current settings and media references when
they are added. For consistent characters, environments, outfits, or visual
style, attach the reference images and choose their roles first, then add the
prompts that should share those references. If you need a different setup, change
the references/settings and add the next batch as separate queue rows.

The queue is saved per project in local browser storage. A running item is reset
to `queued` after a reload so it can be started again. Finished and failed rows
can be cleared from the panel, or `Clear all` can remove every row after a
confirmation warning. If a render has already started, clearing the queue stops
the queue from advancing but does not cancel that in-flight render.

## Performance Notes

Local MLX video generation is slow compared with cloud/API generation, but it
keeps the render local and can leave the Mac usable during generation.

Things that usually help:

- Iterate at 540p or 720p before trying 1080p.
- Use 5-second clips for prompt testing.
- Use `Boost` after a text-to-video prompt works in `Quality`.
- Keep other GPU-heavy apps closed while rendering.
- Keep `.ltx-data` on a fast external SSD if the main drive is low on space.

Things that may hurt quality:

- Pushing 1080p beyond 5 seconds.
- Using `Turbo` for final output.
- Local Mac image-to-video uses a deterministic still-motion fallback. It avoids
  the MLX Q4 distilled I2V tiled-artifact failure, but it is not true generative
  motion. Use the LTX API path for generative image-to-video.
- Raising duration before the prompt/camera motion is stable.

## Environment Variables

The MLX helper recognizes several environment variables:

| Variable | Purpose |
| --- | --- |
| `LTX_APP_DATA_DIR` | App data root. Defaults to `.ltx-data` in development. |
| `LTX_MLX_PATH` | Prepared `ltx-2-mlx` checkout. |
| `LTX_MODELS_DIR` | Model directory. |
| `LTX_MODEL` | MLX Q4 model path passed to the helper. Usually set by the backend. |
| `LTX_GEMMA` | Gemma model path passed to the helper. Usually set by the backend. |
| `LTX_LOW_MEMORY` | Defaults to `true` for the Mac MLX helper. |
| `LTX_IDLE_TIMEOUT` | Helper idle timeout in seconds. Defaults to `1800`. |
| `LTX_ENABLE_MODEL_UPSCALE` | Disabled by default. |
| `LTX_VAE_STREAMING` | Override VAE streaming behavior. Usually leave unset. |

Most users should only set `LTX_APP_DATA_DIR` and `LTX_MLX_PATH` when launching
from a development shell. `LTX_MLX_PATH` should point at
`$PWD/.ltx-data/ltx-2-mlx` after `scripts/setup-mac-mlx.sh` completes.

## Verification Commands

TypeScript:

```bash
pnpm typecheck:ts
```

Focused backend checks:

```bash
cd backend
UV_CACHE_DIR=../.ltx-data/uv-cache ../.venv-tools/bin/uv run --extra test python -m pytest \
  tests/test_settings.py \
  tests/test_generation.py::TestEnhancePromptFlag \
  tests/test_model_download_specs.py \
  tests/test_runtime_policy_decision.py \
  tests/test_models.py \
  -q
```

Regenerate backend OpenAPI types after settings/API schema edits:

```bash
cd backend
UV_CACHE_DIR=../.ltx-data/uv-cache ../.venv-tools/bin/uv run python export_openapi_schema.py
cd ..
corepack pnpm exec openapi-typescript frontend/generated/backend-openapi.json -o frontend/generated/backend-openapi.ts
```

## Troubleshooting

### `LTX_MLX_PATH is not configured` or `No module named ltx_pipelines_mlx`

The MLX runtime was not installed, or the backend cannot see the prepared
`ltx-2-mlx` checkout.

Fix:

```bash
scripts/setup-mac-mlx.sh
export LTX_APP_DATA_DIR="$PWD/.ltx-data"
export LTX_MLX_PATH="$PWD/.ltx-data/ltx-2-mlx"
pnpm dev
```

The backend should resolve the helper Python at:

```text
.ltx-data/ltx-2-mlx/env/bin/python3.11
```

### App still uses the main drive

Development mode should default to `.ltx-data`, but you can force it:

```bash
export LTX_APP_DATA_DIR="$PWD/.ltx-data"
pnpm dev
```

For packaged apps, macOS still defaults to:

```text
~/Library/Application Support/LTXDesktop/
```

unless `LTX_APP_DATA_DIR` is set before launch.

### Longer clips look bad after a few seconds

Use 720p for 10-second local text-to-video clips first; this is the confirmed
working milestone on the 36 GB target Mac. Keep 1080p at 5 seconds until the
longer 1080p path has more validation.

### Generation is slow

That is expected for local MLX video generation. Try:

- `720p`, `5` or `10` seconds while iterating.
- `Boost` speed mode for text-to-video. Image-to-video remains on `Quality`.
- Closing GPU-heavy apps.
- Keeping the project data folder on a fast external SSD.

### MLX setup patch warnings

`scripts/setup-mac-mlx.sh` may print drift warnings for optional I2V or long-clip
boost patches when `ltx-2-mlx` changes shape. The important project-local setup
checks are:

- `.ltx-data/ltx-2-mlx` exists.
- `.ltx-data/ltx-2-mlx/env/bin/python3.11` exists.
- The codec patch and VAE temporal streaming patch apply successfully.

If generation fails with missing `LTX_MLX_PATH`, rerun with the explicit
environment variables shown in the Running section.

### Prompt Enhancer appears to do nothing

Check:

- Settings -> Prompt Enhancer is enabled for the relevant generation type.
- The `gemma-3-12b-it-4bit` folder exists under `.ltx-data/models/`.
- Backend logs show the MLX helper starting and enhancing the prompt.

The enhanced prompt is used internally before text encoding and render; it is
not necessarily shown as a replacement for the text in the prompt box.

### `uv` or model downloads fill the main drive

Set project-local cache variables before setup:

```bash
export LTX_APP_DATA_DIR="$PWD/.ltx-data"
export UV_CACHE_DIR="$PWD/.ltx-data/uv-cache"
export HF_HOME="$PWD/.ltx-data/hf-home"
export XDG_CACHE_HOME="$PWD/.ltx-data/cache"
scripts/setup-mac-mlx.sh
```

### Corrupt virtual environment or I/O errors

If the MLX or backend virtual environment becomes corrupt, recreate it rather
than trying to patch individual files.

For the backend environment:

```bash
cd backend
UV_CACHE_DIR=../.ltx-data/uv-cache ../.venv-tools/bin/uv sync --extra test --extra dev
```

If an external drive reports persistent I/O errors while deleting a corrupt
folder, remount the drive and consider running Disk Utility First Aid before
trying again.

## Git And Repo Hygiene

These paths are intentionally local-only:

- `.ltx-data/`
- model folders
- Hugging Face caches
- generated outputs
- corrupt virtual environment backups such as `backend/.venv-corrupt-*/`

Before committing:

```bash
git status --short
```

Before pushing docs or code changes:

```bash
pnpm typecheck:ts
```

Run focused backend tests when backend settings, model specs, or generation
routing change.
