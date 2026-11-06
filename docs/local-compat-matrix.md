# Local Explore compatibility matrix

Generated from `decide_local_generation_mode`, `decide_video_job`, and the edit RAM floors.
A machine that cannot run a Fast cell is not offered that cell.
Fast rows are decided at the 24 fps / 16:9 anchor: higher rates offer a
subset of these durations, and wider shapes can refuse a kept top cell
(the picker shows it; the budget enforces it).
Retake/Extend rows are viability + RAM floor (`Yes*` — VRAM depends on
your source length): no canonical input exists for a representative
cell, but the floor is input-free. Bands carry no drive class, so
Windows bands take the CPU path (64 GB floor; NVMe maps from disk at
32 GB instead).
Regenerate: `cd backend && uv run python -m runtime_config.local_compat_matrix`.

| Tool | CUDA unsupported | CUDA stream | CUDA full | Windows stream | Windows RAM-gated | Darwin unsupported | macOS below 26 | Darwin stream | Intel Mac / no GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2V | No | Yes | Yes | Yes | No | No | No | Yes | No |
| I2V | No | Yes | Yes | Yes | No | No | No | Yes | No |
| A2V | No | Yes | Yes | Yes | No | No | No | Yes | No |
| LoRAs | No | Yes | Yes | Yes | Yes | No | No | Yes | No |
| IC-LoRA | No | Yes | Yes | No | No | No | No | Yes | No |
| Retake | No | Yes* | Yes* | No | No | No | No | Yes* | No |
| Extend | No | Yes* | Yes* | No | No | No | No | Yes* | No |
| local text encoder | No | Yes | Yes | Yes | Yes | No | No | Yes | No |
| 2.5 download | No | Yes | Yes | Yes | Yes | No | No | Yes | No |

Floors: Windows **9** (fp8 cards; **15** without fp8) / Linux **15** GiB total VRAM (exact bytes, integer fallback);
Windows streamed Fast needs **40** GiB total RAM;
Darwin **32** GiB total RAM and macOS **26** or later
(macOS 14 and 15 run the app but not local generation); Fast cells gate on the MPS pool
(9.75 GiB floor); ~72 GB disk for the 2.5 Fast core pack
(kitchen-sink installs with 2.3, image models, and processors are larger).

CUDA stream (16 GB) still runs T2V/I2V/A2V;
the Fast picker disables 720p/20s and 1080p at 8s and up (greyed out with the reason)
because `decide_video_job` would reject them. CUDA full (31 GB)
keeps those cells (they stream) except 1080p/20s, whose stream ask exceeds `CUDA_FULL_VRAM_GB`.
Darwin stream gates each Fast cell against the sampled MPS
pool (M2/F1 model, 9.75 GiB floor); the 48 GB example keeps the full grid except 1080p/20s
(nominal 0.78 pool ratio — min of 2 F0 probes, SKU-dependent).

Windows stream (16 GB VRAM, 40 GB RAM) runs
the limited grid (540p through 20s, 720p through 10s, 1080p through 6s); 24 GB VRAM runs the
full grid including 1080p/10s, and 12 GB VRAM runs 540p through 10s plus 720p
through 8s (no 1080p — the 5s cell needs 14.14 reserved). Below 40 GB RAM, streamed Fast cells are not
offered (`ram_shortfall`). Streamed edit cells gate on E3-measured floors (64 GB CPU,
32 GB NVMe DISK); streamed IC-LoRA on L5-measured floors (64 GB CPU,
48 GB DISK, L5-48 holds both temps). The live picker lists every total-fitting cell;
GenSpace refuses cells over free VRAM at click, and queued jobs re-check at execute.
On NVMe drives, streamed Fast maps weights from disk and the RAM floor is **32** GiB instead.

1440p and 4K Fast cells run long clips in overlapping time windows, and the job budget prices one window.
1440p and 4K are offered for Fast, A2V and the Home recipes; Retake, Extend and IC-LoRA need `highResVideoToVideo`. Both are provisional, and
the Mac window extra is a copy of the Windows one. The cells above are the 24 fps / 16:9 anchor and do not list these two rows.

Footnote: Darwin IC-LoRA gates each duration against the sampled MPS
pool (I1/I2 model, 9.84 GiB floor, growth line
P20-refit on 1.4.1: 5.75 + 6.105e-4 per token, +0.50 for
ingredients); the 48 GB example keeps 20s portrait.
