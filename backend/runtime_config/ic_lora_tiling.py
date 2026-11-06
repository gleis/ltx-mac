"""Spatial transformer tiling of a catalog IC-LoRA (Restore, Refine Details).

These LoRAs trained on one tile size. Their cards say to run every output size in
overlapping tiles of that size, merged inside each denoising step, even at full HD. A
single untiled pass is outside the training bucket. ltx-pipelines runs this as
``ICLoraStageConfig.tiling`` (``FixedSizeSpatialTiling``, overlap 0.5 at most).

Tiling is a property of the LoRA, so it lives in the catalog ``default_settings``. The job
budget reads the tile size as the size of the transformer pass. The spatial cap still
reads the output size.

No imports from other runtime modules, so ``api_types`` and the budget can import it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The overlap fraction we request. It is the most upstream accepts, and the least the
# Refine Details card allows ("never lower the overlap below 0.5"). It is a request, not
# the layout: upstream ``split_by_size_pinned`` still caps the tile count so that no cell
# is covered by three tiles. On a 1920x1088 canvas Restore's 960x544 tiles lay out 3x2,
# Refine's 1024x576 tiles 2x2. The count does not bind the peak: tiles run sequentially,
# so one tile plus the canvas decode is the whole ask — six tiles peak at tile + 1.28,
# not at six tiles.
IC_LORA_TILE_OVERLAP = 0.5

_TILE_GRID = 32  # pixel multiple of the transformer latent grid
_FRAME_GRID = 8  # pixel frames per latent frame; windows are 8k + 1 frames
# ltx-pipelines refuses a window carry shorter than three latent frames (17 pixel frames).
_MIN_CARRY_FRAMES = 2 * _FRAME_GRID + 1
# ltx-pipelines default carry, kept by a tiling that sets a window and no carry.
DEFAULT_CARRY_FRAMES = 25


class IcLoraTiling(BaseModel):
    """Tile size of a catalog IC-LoRA, and an optional fixed temporal window.

    ``long_side`` and ``short_side`` are given for a landscape canvas. A portrait canvas
    swaps them (Refine Details card: "1024x576, or 576x1024 portrait").

    ``window_frames`` is set when the LoRA drifts over a longer clip (Restore: past 97
    frames detail is lost and colour drifts). A clip longer than the window always runs
    in windows of this length. None leaves the choice to the job budget.

    ``carry_frames`` is the part of a window that the next window re-renders and pins at
    its start. It needs a ``window_frames``. None keeps ``DEFAULT_CARRY_FRAMES``.

    ``max_output_long_side`` and ``max_output_short_side`` raise the spatial cap of the
    job for this LoRA. The transformer sees one tile at a time, so the output can be
    larger than the local 1080p cell. None keeps that cell.
    """

    model_config = ConfigDict(strict=True, frozen=True)
    long_side: int = Field(ge=64)
    short_side: int = Field(ge=64)
    window_frames: int | None = Field(default=None, ge=9)
    carry_frames: int | None = Field(default=None, ge=_MIN_CARRY_FRAMES)
    max_output_long_side: int | None = Field(default=None, ge=64)
    max_output_short_side: int | None = Field(default=None, ge=64)

    @model_validator(mode="after")
    def _check_grid(self) -> "IcLoraTiling":
        if self.long_side % _TILE_GRID or self.short_side % _TILE_GRID:
            raise ValueError(f"tile sides must be multiples of {_TILE_GRID}")
        if self.short_side > self.long_side:
            raise ValueError("short_side must not exceed long_side")
        if self.window_frames is not None and (self.window_frames - 1) % _FRAME_GRID:
            raise ValueError(f"window_frames must be {_FRAME_GRID}k + 1")
        if self.carry_frames is not None:
            if self.window_frames is None:
                raise ValueError("carry_frames needs window_frames")
            if (self.carry_frames - 1) % _FRAME_GRID:
                raise ValueError(f"carry_frames must be {_FRAME_GRID}k + 1")
            if self.carry_frames >= self.window_frames:
                raise ValueError("carry_frames must be below window_frames")
        if self.window_frames is not None:
            # A middle window keeps window - carry frames. Upstream cross-fades the whole
            # carry, and it needs the cross-fade to be shorter than the kept frames. A
            # window without its own carry runs with the default one.
            carry = DEFAULT_CARRY_FRAMES if self.carry_frames is None else self.carry_frames
            if self.window_frames <= 2 * carry:
                raise ValueError("window_frames must be more than twice carry_frames")
        if (self.max_output_long_side is None) != (self.max_output_short_side is None):
            raise ValueError("set both max_output sides or neither")
        if (
            self.max_output_long_side is not None
            and self.max_output_short_side is not None
            and self.max_output_short_side > self.max_output_long_side
        ):
            raise ValueError("max_output_short_side must not exceed max_output_long_side")
        return self

    def output_cap(self) -> tuple[int, int] | None:
        """(long, short) cap of the output size, or None for the local 1080p cell."""
        if self.max_output_long_side is None or self.max_output_short_side is None:
            return None
        return self.max_output_long_side, self.max_output_short_side

    def tile_size(self, width: int, height: int) -> tuple[int, int]:
        """(width, height) of one tile on a canvas of this size.

        The tile follows the canvas orientation. An axis shorter than its tile stays
        untiled, so the tile never exceeds the canvas.
        """
        if height > width:
            tile_width, tile_height = self.short_side, self.long_side
        else:
            tile_width, tile_height = self.long_side, self.short_side
        return min(width, tile_width), min(height, tile_height)


def transformer_pass_size(width: int, height: int, tiling: IcLoraTiling | None) -> tuple[int, int]:
    """Size of one transformer call on a canvas. The tile when tiled, else the canvas."""
    if tiling is None:
        return width, height
    return tiling.tile_size(width, height)
