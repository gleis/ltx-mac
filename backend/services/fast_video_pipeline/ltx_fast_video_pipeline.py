"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from contextlib import nullcontext
import os
from typing import TYPE_CHECKING, Final, Literal

import torch

from api_types import ImageConditioningInput
from frame_math import AutoDurationSpec
from services.fast_video_pipeline.windowed_run import encode_result, measured_windows, plan_windows
from services.ic_lora_pipeline.ic_lora_inference import InferenceResult
from services.ltx_pipeline_common import (
    auto_tiling_config,
    build_model_paths,
)
from services.services_utils import PipelineTilingType, device_supports_fp8

if TYPE_CHECKING:
    from ltx_pipelines.chunks import ChunkConfig

if TYPE_CHECKING:
    from ltx_pipelines.utils.types import OffloadMode


class LTXFastVideoPipeline:
    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        loras: list[tuple[str, float]] | None = None,
        *,
        offload_mode: OffloadMode,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
    ) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            loras=loras or [],
            offload_mode=offload_mode,
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
            duration_head_path=duration_head_path,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        loras: list[tuple[str, float]] | None = None,
        *,
        offload_mode: OffloadMode,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
    ) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
        from services.fast_video_pipeline.distilled_skip_stage_2 import DistilledPipelineWithSkipStage2

        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._video_vae_path = video_vae_path
        self._audio_vae_path = audio_vae_path
        self._duration_head_path = duration_head_path
        self._device = device
        self._offload_mode = offload_mode
        self._quantization = build_fp8_cast_policy(checkpoint_path) if device_supports_fp8(device) else None
        self._loras = loras or []

        lora_entries = [
            LoraPathStrengthAndSDOps(path=path, strength=scale, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            for path, scale in self._loras
        ]

        self.pipeline = DistilledPipelineWithSkipStage2(
            model_paths=build_model_paths(
                checkpoint_path,
                gemma_root,
                video_vae_path=video_vae_path,
                audio_vae_path=audio_vae_path,
                duration_head_path=duration_head_path,
            ),
            spatial_upsampler_path=upsampler_path,
            loras=lora_entries,
            device=device,
            quantization=self._quantization,
            offload_mode=self._offload_mode,
        )

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int | AutoDurationSpec,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: PipelineTilingType,
        *,
        skip_stage_2: bool = False,
        chunk_config: ChunkConfig | None = None,
    ) -> InferenceResult:
        """Run the pipeline. A windowed job streams, so the caller must encode inside the
        keyframe-guiding scope and the progress scope, and close the streams after."""
        from ltx_pipelines.utils.types import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.types import AutoDuration

        pipeline_num_frames: int | AutoDuration = (
            AutoDuration(min_seconds=num_frames.min_seconds, max_seconds=num_frames.max_seconds)
            if isinstance(num_frames, AutoDurationSpec)
            else num_frames
        )
        ltx_images = [_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images]
        if chunk_config is None:
            result = self.pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=pipeline_num_frames,
                frame_rate=frame_rate,
                images=ltx_images,
                tiling_config=tiling_config,
                skip_stage_2=skip_stage_2,
            )
            return InferenceResult(result.video, result.audio, result.num_frames, result.tiling_config)
        if skip_stage_2:
            raise ValueError("A windowed job runs both stages")
        from ltx_pipelines.chunks import split_decoded_chunks

        chunks, resolved_frames, resolved_tiling = self.pipeline.stream_chunks(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=pipeline_num_frames,
            frame_rate=frame_rate,
            images=ltx_images,
            tiling_config=tiling_config,
            chunk_config=chunk_config,
        )
        video, audio, audio_sampling_rate = split_decoded_chunks(chunks)
        streams = (video, chunks) if audio is None else (video, audio, chunks)
        return InferenceResult(video, audio, resolved_frames, resolved_tiling, audio_sampling_rate, streams)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int | AutoDurationSpec,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
        speed_mode: Literal["quality", "boost", "turbo"] = "quality",
        *,
        guide_all_images: bool = False,
        skip_stage_2: bool = False,
        chunk_pixel_frames: int | None = None,
    ) -> None:
        del speed_mode
        from services.denoising_progress import distilled_total_steps, track_denoising
        from services.fast_video_pipeline.distilled_keyframe_guiding import distilled_keyframe_guiding

        chunk_config, chunk_count = plan_windows(
            chunk_pixel_frames, width=width, height=height, num_frames=num_frames, frame_rate=frame_rate
        )
        steps = distilled_total_steps(stage_2=not skip_stage_2) * chunk_count
        guiding = distilled_keyframe_guiding() if guide_all_images else nullcontext()
        measured = measured_windows(
            chunk_config, width=width, height=height, window_count=chunk_count,
            num_frames=num_frames, frame_rate=frame_rate, device=self._device,
        )
        with track_denoising(steps), guiding, measured:
            result = self._run_inference(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=auto_tiling_config(),
                skip_stage_2=skip_stage_2,
                chunk_config=chunk_config,
            )
            encode_result(result, fps=int(frame_rate), output_path=output_path, streamed=chunk_config is not None)

    def generate_reference_i2v(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        lora_path: str,
        output_path: str,
    ) -> None:
        del prompt, seed, height, width, num_frames, frame_rate, images, video_conditioning, lora_path, output_path
        raise RuntimeError("Multi-reference local generation is only supported by the Mac MLX pipeline")

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9

        try:
            result = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=auto_tiling_config(),
            )
            encode_result(result, fps=8, output_path=output_path, streamed=False)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def enhance_prompt(self, prompt: str, *, mode: Literal["t2v", "i2v"], seed: int) -> str:
        del mode, seed
        return prompt

    def compile_transformer(self) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.model.transformer.compiling import CompilationConfig
        from services.fast_video_pipeline.distilled_skip_stage_2 import DistilledPipelineWithSkipStage2

        lora_entries = [
            LoraPathStrengthAndSDOps(path=path, strength=scale, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            for path, scale in self._loras
        ]

        self.pipeline = DistilledPipelineWithSkipStage2(
            model_paths=build_model_paths(
                self._checkpoint_path,
                self._gemma_root,
                video_vae_path=self._video_vae_path,
                audio_vae_path=self._audio_vae_path,
                duration_head_path=self._duration_head_path,
            ),
            spatial_upsampler_path=self._upsampler_path,
            loras=lora_entries,
            device=self._device,
            quantization=self._quantization,
            offload_mode=self._offload_mode,
            compilation_config=CompilationConfig(),
        )
