"""Fast video pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    import torch


class FastVideoPipeline(Protocol):
    pipeline_kind: ClassVar[Literal["fast"]]

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
    ) -> "FastVideoPipeline":
        ...

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
        speed_mode: Literal["quality", "boost", "turbo"] = "quality",
    ) -> None:
        ...

    def warmup(self, output_path: str) -> None:
        ...

    def enhance_prompt(self, prompt: str, *, mode: Literal["t2v", "i2v"], seed: int) -> str:
        ...

    def compile_transformer(self) -> None:
        ...
