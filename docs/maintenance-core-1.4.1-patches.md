# Private patch review for LTX core/pipelines 1.4.1

Reviewed against the locked upstream commit
`2d6e71c88be37b55a2dd698c2dff447edfbe5898`, when integrating Desktop v1.3.1.
This records source compatibility and the reason each startup patch remains.
It does not establish CUDA hardware performance or model quality.

| Startup patch | Upstream boundary inspected | Decision |
| --- | --- | --- |
| `safetensors_loader_fix` | `ltx_core/loader/sft_loader.py`, `SafetensorsStateDictLoader.load` still opens tensor shards with `safe_open` | Retain the direct read loader. The upstream mmap path remains. |
| `disk_streaming_pread` | `ltx_core/block_streaming/disk.py`, `DiskTensorReader.__init__` does not select a read backend | Retain Darwin-only `pread`; preserve Windows mmap behavior. |
| `linux_disk_streaming_backend` | The same reader retains the default mmap backend on Linux | Retain the conditional commit-headroom fallback; it remains a no-op off Linux. |
| `safetensors_metadata_fix` | Loader metadata, IC-LoRA downscale/version detection, Gemma model ID, FP8 scales and streaming key scans still use the patched private boundaries | Retain. Startup asserts verify the bound symbols still exist, including the public IC-LoRA function names. |
| `pinned_pool_fix` | `ltx_core/block_streaming/utils.py`, `alloc_buffer` falls back from failed host registration to pinned `torch.empty` | Retain the host-allocation error handling and CUDA sticky-error cleanup. |
| `diffusion_stage_cache` | `ltx_pipelines/utils/blocks.py`, `DiffusionStage._transformer_ctx` still builds through `gpu_model` on each non-streaming call | Retain the opt-in, generation-scoped cache; streaming bypass and eviction remain required. |
| `diffvae_decode_vram` | `VideoDecoder.__call__` builds its decoder with the supplied tiling configuration; it does not evict Desktop's transformer cache or re-plan against the post-eviction budget | Retain the Desktop eviction and tile clamps. This Torch patch is separate from the MLX decoder fix. |
| `natten_libnatten_gate` | `ltx_core/model/video_vae/transformer/attention.py`, `natten_available` returns import availability alone | Retain the `HAS_LIBNATTEN` gate for installations without the native extension. |
| `diffusion_interrupt` | `DiffusionStage.__call__` still accepts `denoiser` as its first parameter after `self`, with no interruption callback | Retain cancellation/progress wrapping. |
| `mps_sdpa_torch` | Desktop installs its MPS attention adapter after the Torch attention setup; native Diffusers still calls Torch SDPA | Retain the explicit MPS adapter. Package the zero-copy extension against the shipped Python/Torch pair. |

The old `record_stream_fix` is removed because upstream replaced
`LayerStreamingWrapper` with block streaming. The new `StreamSync` and
`BufferPool` coordinate copy/compute and wait for completion events before
buffer reuse. There are no `record_stream` calls in this locked core source.
Reintroducing the old private-class patch would fail against the new API.

Validation includes real backend startup on Apple Silicon with the locked
packages and authenticated health/runtime-policy requests. The integrated
macOS, Linux and Windows Python CI jobs passed at `62dce14`; the remaining
workflow failure at that revision was the unchanged upstream audio fixture's
repository size check. Hardware generation and packaged-runtime checks are
recorded separately and must not be inferred from the unit-test results.
