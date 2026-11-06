"""Fast video pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

from api_types import ImageConditioningInput
from frame_math import AutoDurationSpec

if TYPE_CHECKING:
    import torch
    from ltx_pipelines.utils.types import OffloadMode


class FastVideoPipeline(Protocol):
    pipeline_kind: ClassVar[Literal["fast"]]

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
    ) -> "FastVideoPipeline":
        ...

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
        ...

    def generate_reference_i2v(
        self, prompt: str, seed: int, height: int, width: int, num_frames: int,
        frame_rate: float, images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]], lora_path: str, output_path: str,
    ) -> None:
        ...

    def warmup(self, output_path: str) -> None:
        ...

    def enhance_prompt(self, prompt: str, *, mode: Literal["t2v", "i2v"], seed: int) -> str:
        ...

    def compile_transformer(self) -> None:
        ...
