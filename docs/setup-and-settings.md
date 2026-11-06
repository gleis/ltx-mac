# Setup and settings

## Platform requirements

These values match the [LTX Desktop page](https://ltx.io/ltx-desktop). All platforms need **100 GB+ free disk space**, up to **160 GB+** with additional LoRAs and IC-LoRAs.

### Windows (local generation)

- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA, **24 GB+ VRAM** (32 GB recommended)
- 32 GB RAM

### Linux (local generation)

- Ubuntu 22.04+ or similar distro (x64 or arm64)
- NVIDIA GPU with CUDA and the NVIDIA driver installed (PyTorch bundles the CUDA runtime), **24 GB+ VRAM** (32 GB recommended)
- 32 GB RAM

### macOS (local generation)

- macOS 26+ (Tahoe) for local generation. The app itself needs macOS 14 or later. On macOS 14 and 15 the app shows the **Coming soon to your device** notice.
- Apple Silicon (M1 or later) with **48 GB+**: full local generation
- Apple Silicon with 32 GB: limited (540p and 5 s only)
- Apple Silicon with less than 24 GB, or an Intel Mac: Gen Space API only. Intel Macs have no MPS backend.

### Lower-spec machines

The app offers local generation on lower-spec machines, but we do not recommend them. The app uses these floors (see `backend/runtime_config/runtime_policy.py` and `docs/local-compat-matrix.md`):

- CUDA: **9 GiB VRAM** on Windows with an fp8-capable GPU, **15 GiB** otherwise (Linux, non-fp8 Windows; integer GiB from `total_memory // 1024**3`)
- macOS: **32 GiB total RAM** (machine SKU, not free RAM at launch) and **macOS 26 or later**

Below a floor, local generation is unavailable and the app shows the **Coming soon to your device** notice. An LTX API key is not required.

## First run & data locations

LTX Desktop stores app data (settings, models, logs) in:

- **Windows:** `%LOCALAPPDATA%\LTXDesktop\`
- **macOS:** `~/Library/Application Support/LTXDesktop/`
- **Linux:** `$XDG_DATA_HOME/LTXDesktop/` (default: `~/.local/share/LTXDesktop/`)

Model weights are downloaded into the `models/` subfolder (this can be large and may take time).

On first launch, the Setup screen guides you through the steps below. It requires internet.

1. Sign in to Hugging Face and accept the license terms of [gated models](models-and-loras.md#gated-models) (the 2.5 weights are gated).
2. Optionally add an LTX API key for free cloud text encoding.
3. Choose the models to download and the models folder.
4. Optionally select **Opt out of basic analytics**.

The app shows a splash screen while it installs its Python environment.

Text encoding: to generate videos you must configure text encoding:

- **LTX API key** (cloud text encoding) — **text encoding via the API is completely FREE** and highly recommended to speed up inference and save memory. Generate a free API key at the [LTX Console](https://console.ltx.video/). [Read more](https://ltx.io/model/model-blog/ltx-2-better-control-for-real-workflows).
- **Local Text Encoder** (extra download; enables fully-local operation on supported hardware) — if you don't wish to generate an API key, you can download the local encoder in **Settings > Models** and select it in **Settings > General > Text encoding**.

## Settings

Open **Settings** from the bottom of the sidebar.

| Tab | What it controls |
| --- | --- |
| **General** | Appearance (theme: system, light, or dark); auto-enhance for local and API generation; fast decode (faster video decode with slightly lower fidelity); text encoding (LTX API or local encoder); NVIDIA performance options (Torch Compile and diffusion stage cache, experimental); anonymous analytics |
| **Models** | Download and manage checkpoints, LoRAs, and encoder models. See [Managing models](models-and-loras.md#managing-models). Hidden in API-only mode |
| **Legacy** | Options for Gen Space and the Video Editor: project assets folder, default base model, preference for cloud video and image generation, and seed lock |
| **API Keys** | Optional LTX, fal, and Gemini keys. The app checks a key before it saves it, and you can remove a saved key |
| **About** | App version, the automatic update switch, and manual update checks |
