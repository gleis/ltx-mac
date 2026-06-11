"""Video generation orchestration handler."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from PIL import Image

from api_types import (
    GenerateVideoCancelledResponse,
    GenerateVideoCompleteResponse,
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    ImageConditioningInput,
    VideoCameraMotion,
)
from _routes._errors import HTTPError
from api_model_specs import (
    build_generate_video_model_specs_response,
    validate_generate_video_request,
)
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from server_utils.media_validation import (
    normalize_optional_path,
    validate_audio_file,
    validate_image_file,
)
from runtime_config.model_download_specs import get_existing_cp_path
from services.interfaces import LTXAPIClient
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from state.app_state_types import AppState
from state.app_settings import should_video_generate_with_ltx_api

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

FORCED_API_MODEL_MAP: dict[str, str] = {
    "fast": "ltx-2-3-fast",
    "pro": "ltx-2-3-pro",
}
FORCED_API_RESOLUTION_MAP: dict[str, dict[str, str]] = {
    "1080p": {"16:9": "1920x1080", "9:16": "1080x1920"},
    "1440p": {"16:9": "2560x1440", "9:16": "1440x2560"},
    "2160p": {"16:9": "3840x2160", "9:16": "2160x3840"},
}
FORCED_API_ALLOWED_ASPECT_RATIOS = {"16:9", "9:16"}
_LTX_INSUFFICIENT_FUNDS_MESSAGE = "Your LTX API credits are insufficient for this generation. Buy more credits and try again."
_IC_LORA_UNION_CONTROL_CP_ID = "ltx-2.3-22b-ic-lora-union-control-ref0.5"


class VideoGenerationHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        ltx_api_client: LTXAPIClient,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._ltx_api_client = ltx_api_client

    def get_model_specs(self) -> GenerateVideoModelsSpecsResponse:
        return build_generate_video_model_specs_response(self.config.local_generations_mode)

    def generate(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        use_api_specs = should_video_generate_with_ltx_api(
            force_api_generations=self.config.force_api_generations,
            settings=self.state.app_settings,
        )
        validation_error = validate_generate_video_request(
            req,
            use_api_specs=use_api_specs,
            local_generations_mode=self.config.local_generations_mode,
        )
        if validation_error is not None:
            raise HTTPError(422, validation_error, code="INVALID_VIDEO_GENERATION_SPEC")

        if use_api_specs:
            return self._generate_forced_api(req)

        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        resolution = req.resolution
        duration = req.duration
        fps = req.fps

        audio_path = normalize_optional_path(req.audioPath)
        if audio_path:
            if self.config.local_generations_mode == "mac_mlx_q4":
                raise HTTPError(400, "LOCAL_MAC_MLX_A2V_NOT_SUPPORTED_IN_FIRST_MILESTONE")
            return self._generate_a2v(req, duration, fps, audio_path=audio_path)

        logger.info("Resolution %s - using fast pipeline", resolution)

        RESOLUTION_MAP_16_9: dict[str, tuple[int, int]] = {
            "540p": (960, 544),
            "720p": (1280, 704),
            "1080p": (1920, 1088),
        }

        def get_16_9_size(res: str) -> tuple[int, int]:
            size = RESOLUTION_MAP_16_9.get(res)
            if size is None:
                raise HTTPError(400, "INVALID_LOCAL_RESOLUTION")
            return size

        def get_9_16_size(res: str) -> tuple[int, int]:
            w, h = get_16_9_size(res)
            return h, w

        match req.aspectRatio:
            case "9:16":
                width, height = get_9_16_size(resolution)
            case "16:9":
                width, height = get_16_9_size(resolution)

        num_frames = self._compute_num_frames(duration, fps)

        image = None
        image_path = normalize_optional_path(req.imagePath)
        if image_path:
            image = self._prepare_image(image_path, width, height)
            logger.info("Image: %s -> %sx%s", image_path, width, height)
        reference_image_paths = [
            str(validate_image_file(path))
            for path in req.referenceImagePaths
            if normalize_optional_path(path) is not None
        ]

        generation_id = self._make_generation_id()
        seed = self._resolve_seed()

        try:
            self._pipelines.load_gpu_pipeline("fast")
            self._generation.start_generation(generation_id)
            if len(reference_image_paths) > 1:
                if self.config.local_generations_mode != "mac_mlx_q4":
                    raise HTTPError(400, "LOCAL_REFERENCE_I2V_REQUIRES_MAC_MLX_Q4")
                output_path = self._generate_local_reference_i2v(
                    prompt=req.prompt,
                    reference_image_paths=reference_image_paths,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    seed=seed,
                    camera_motion=req.cameraMotion,
                )
                self._generation.complete_generation(output_path)
                return GenerateVideoCompleteResponse(status="complete", video_path=output_path)

            if image is not None and self.config.local_generations_mode == "mac_mlx_q4":
                output_path = self._generate_local_i2v_still_motion(
                    image=image,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                )
                self._generation.complete_generation(output_path)
                return GenerateVideoCompleteResponse(status="complete", video_path=output_path)

            output_path = self.generate_video(
                prompt=req.prompt,
                image=image,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
                camera_motion=req.cameraMotion,
                negative_prompt=req.negativePrompt,
            )

            self._generation.complete_generation(output_path)
            return GenerateVideoCompleteResponse(status="complete", video_path=output_path)

        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Generation cancelled by user")
                return GenerateVideoCancelledResponse(status="cancelled")

            raise HTTPError(500, str(e)) from e

    def generate_video(
        self,
        prompt: str,
        image: Image.Image | None,
        height: int,
        width: int,
        num_frames: int,
        fps: float,
        seed: int,
        camera_motion: VideoCameraMotion,
        negative_prompt: str,
    ) -> str:
        t_total_start = time.perf_counter()
        gen_mode = "i2v" if image is not None else "t2v"
        logger.info("[%s] Generation started (model=fast, %dx%d, %d frames, %d fps)", gen_mode, width, height, num_frames, int(fps))

        if self._generation.is_generation_cancelled():
            raise RuntimeError("Generation was cancelled")

        total_steps = 8

        self._generation.update_progress("loading_model", 5, 0, total_steps)
        t_load_start = time.perf_counter()
        pipeline_state = self._pipelines.load_gpu_pipeline("fast")
        t_load_end = time.perf_counter()
        logger.info("[%s] Pipeline load: %.2fs", gen_mode, t_load_end - t_load_start)

        enhanced_prompt = prompt + self.config.camera_motion_prompts.get(camera_motion, "")

        images: list[ImageConditioningInput] = []
        temp_image_path: str | None = None
        if image is not None:
            temp_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            image.save(temp_image_path)
            images = [ImageConditioningInput(path=temp_image_path, frame_idx=0, strength=1.0)]

        output_path = self._make_output_path()

        try:
            settings = self.state.app_settings
            use_api_encoding = not self._text.should_use_local_encoding()
            if image is not None:
                prompt_enhancer_enabled = settings.prompt_enhancer_enabled_i2v
            else:
                prompt_enhancer_enabled = settings.prompt_enhancer_enabled_t2v

            if (
                self.config.local_generations_mode == "mac_mlx_q4"
                and prompt_enhancer_enabled
            ):
                self._generation.update_progress("enhancing_prompt", 8, 0, total_steps)
                t_enhance_start = time.perf_counter()
                enhanced_prompt = pipeline_state.pipeline.enhance_prompt(enhanced_prompt, mode=gen_mode, seed=seed)
                t_enhance_end = time.perf_counter()
                logger.info("[%s] Prompt enhancement (local MLX): %.2fs", gen_mode, t_enhance_end - t_enhance_start)

            enhance = use_api_encoding and prompt_enhancer_enabled

            encoding_method = "api" if use_api_encoding else "local"
            self._generation.update_progress("encoding_text", 10, 0, total_steps)
            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(enhanced_prompt, enhance_prompt=enhance)
            t_text_end = time.perf_counter()
            logger.info("[%s] Text encoding (%s): %.2fs", gen_mode, encoding_method, t_text_end - t_text_start)

            self._generation.update_progress("inference", 15, 0, total_steps)

            height = round(height / 64) * 64
            width = round(width / 64) * 64

            t_inference_start = time.perf_counter()
            speed_mode = "quality"
            if self.config.local_generations_mode == "mac_mlx_q4" and gen_mode == "t2v":
                speed_mode = settings.local_mlx_speed_mode

            pipeline_state.pipeline.generate(
                prompt=enhanced_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                images=images,
                output_path=str(output_path),
                speed_mode=speed_mode,
            )
            t_inference_end = time.perf_counter()
            logger.info("[%s] Inference: %.2fs", gen_mode, t_inference_end - t_inference_start)

            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise RuntimeError("Generation was cancelled")

            t_total_end = time.perf_counter()
            logger.info("[%s] Total generation: %.2fs (load=%.2fs, text=%.2fs, inference=%.2fs)",
                        gen_mode, t_total_end - t_total_start,
                        t_load_end - t_load_start, t_text_end - t_text_start, t_inference_end - t_inference_start)

            self._generation.update_progress("complete", 100, total_steps, total_steps)
            return str(output_path)
        finally:
            self._text.clear_api_embeddings()
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    def _generate_a2v(
        self, req: GenerateVideoRequest, duration: int, fps: int, *, audio_path: str
    ) -> GenerateVideoResponse:
        validated_audio_path = validate_audio_file(audio_path)
        audio_path_str = str(validated_audio_path)

        RESOLUTION_MAP: dict[str, tuple[int, int]] = {
            "540p": (960, 576),
            "720p": (1280, 704),
            "1080p": (1920, 1088),
        }
        size = RESOLUTION_MAP.get(req.resolution)
        if size is None:
            raise HTTPError(400, "INVALID_LOCAL_A2V_RESOLUTION")
        width, height = size
        if req.aspectRatio == "9:16":
            width, height = height, width

        num_frames = self._compute_num_frames(duration, fps)

        image = None
        temp_image_path: str | None = None
        image_path = normalize_optional_path(req.imagePath)
        if image_path:
            image = self._prepare_image(image_path, width, height)

        seed = self._resolve_seed()

        generation_id = self._make_generation_id()

        try:
            a2v_state = self._pipelines.load_a2v_pipeline()
            self._generation.start_generation(generation_id)

            enhanced_prompt = req.prompt + self.config.camera_motion_prompts.get(req.cameraMotion, "")
            neg = req.negativePrompt if req.negativePrompt else self.config.default_negative_prompt

            images: list[ImageConditioningInput] = []
            if image is not None:
                temp_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                image.save(temp_image_path)
                images = [ImageConditioningInput(path=temp_image_path, frame_idx=0, strength=1.0)]

            output_path = self._make_output_path()

            total_steps = 11  # distilled: 8 steps (stage 1) + 3 steps (stage 2)

            a2v_settings = self.state.app_settings
            a2v_use_api = not self._text.should_use_local_encoding()
            if image is not None:
                a2v_enhance = a2v_use_api and a2v_settings.prompt_enhancer_enabled_i2v
            else:
                a2v_enhance = a2v_use_api and a2v_settings.prompt_enhancer_enabled_t2v

            self._generation.update_progress("loading_model", 5, 0, total_steps)
            self._generation.update_progress("encoding_text", 10, 0, total_steps)
            self._text.prepare_text_encoding(enhanced_prompt, enhance_prompt=a2v_enhance)
            self._generation.update_progress("inference", 15, 0, total_steps)

            a2v_state.pipeline.generate(
                prompt=enhanced_prompt,
                negative_prompt=neg,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=total_steps,
                images=images,
                audio_path=audio_path_str,
                audio_start_time=0.0,
                audio_max_duration=None,
                output_path=str(output_path),
            )

            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise RuntimeError("Generation was cancelled")

            self._generation.update_progress("complete", 100, total_steps, total_steps)
            self._generation.complete_generation(str(output_path))
            return GenerateVideoCompleteResponse(status="complete", video_path=str(output_path))

        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Generation cancelled by user")
                return GenerateVideoCancelledResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e
        finally:
            self._text.clear_api_embeddings()
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    def _prepare_image(self, image_path: str, width: int, height: int) -> Image.Image:
        validated_path = validate_image_file(image_path)
        try:
            img = Image.open(validated_path).convert("RGB")
        except Exception:
            raise HTTPError(400, f"Invalid image file: {image_path}") from None
        img_w, img_h = img.size
        target_ratio = width / height
        img_ratio = img_w / img_h
        if img_ratio > target_ratio:
            new_h = height
            new_w = int(img_w * (height / img_h))
        else:
            new_w = width
            new_h = int(img_h * (width / img_w))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return resized.crop((left, top, left + width, top + height))

    def _generate_local_i2v_still_motion(
        self,
        *,
        image: Image.Image,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
    ) -> str:
        """Render a deterministic local I2V fallback while MLX Q4 I2V is unstable."""
        self._generation.update_progress("inference", 15, 0, 1)
        output_path = self._make_output_path()
        logger.warning(
            "Mac local MLX image-to-video uses still-motion fallback because distilled Q4 I2V collapses into tiled artifacts"
        )

        try:
            import imageio_ffmpeg  # type: ignore[reportMissingTypeStubs]

            ffmpeg = str(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception as exc:
            raise HTTPError(500, "Bundled ffmpeg is required for local image-to-video fallback") from exc

        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-frames:v",
            str(num_frames),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            total = max(num_frames - 1, 1)
            for frame_idx in range(num_frames):
                if self._generation.is_generation_cancelled():
                    raise RuntimeError("Generation was cancelled")
                t = frame_idx / total
                zoom = 1.0 + 0.035 * t
                crop_w = max(1, int(width / zoom))
                crop_h = max(1, int(height / zoom))
                # Tiny diagonal drift keeps the shot alive without inventing content.
                x_bias = int((width - crop_w) * (0.5 + 0.12 * t))
                y_bias = int((height - crop_h) * (0.5 - 0.08 * t))
                left = min(max(x_bias, 0), width - crop_w)
                top = min(max(y_bias, 0), height - crop_h)
                frame = image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.LANCZOS)
                proc.stdin.write(frame.tobytes())
                if frame_idx % max(fps, 1) == 0:
                    progress = 15 + int(80 * frame_idx / total)
                    self._generation.update_progress("inference", progress, frame_idx, num_frames)
            proc.stdin.close()
            return_code = proc.wait(timeout=120)
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()

        if return_code != 0:
            raise HTTPError(500, f"Local image-to-video fallback failed: {stderr.strip()}")
        self._generation.update_progress("complete", 100, num_frames, num_frames)
        return str(output_path)

    def _generate_local_reference_i2v(
        self,
        *,
        prompt: str,
        reference_image_paths: list[str],
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
        camera_motion: VideoCameraMotion,
    ) -> str:
        self._generation.update_progress("loading_model", 5, 0, 12)
        try:
            lora_path = get_existing_cp_path(
                self.models_dir,
                _IC_LORA_UNION_CONTROL_CP_ID,
            )
        except FileNotFoundError as exc:
            raise HTTPError(
                409,
                "Local multi-reference generation requires the IC-LoRA Union Control model. Download it from Settings > Models, then try again.",
                code="LOCAL_REFERENCE_I2V_MODEL_MISSING",
            ) from exc

        prepared_paths: list[str] = []
        control_path: str | None = None
        try:
            prepared_images: list[Image.Image] = [
                self._prepare_image(path, width, height)
                for path in reference_image_paths
            ]
            frame_indices = self._reference_frame_indices(len(prepared_images), num_frames)
            for image in prepared_images:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()
                image.save(tmp.name)
                prepared_paths.append(tmp.name)

            self._generation.update_progress("encoding_text", 10, 0, 12)
            pipeline_state = self._pipelines.load_gpu_pipeline("fast")
            enhanced_prompt = prompt + self.config.camera_motion_prompts.get(camera_motion, "")
            if self.state.app_settings.prompt_enhancer_enabled_i2v:
                self._generation.update_progress("enhancing_prompt", 12, 0, 12)
                enhanced_prompt = pipeline_state.pipeline.enhance_prompt(enhanced_prompt, mode="i2v", seed=seed)

            self._generation.update_progress("preparing_references", 14, 0, 12)
            control_path = self._make_reference_control_video(
                prepared_images=prepared_images,
                width=width,
                height=height,
                num_frames=self._reference_control_frame_count(num_frames),
                fps=fps,
            )
            anchors = [
                ImageConditioningInput(path=path, frame_idx=frame_idx, strength=1.0)
                for path, frame_idx in zip(prepared_paths, frame_indices, strict=True)
            ]

            output_path = self._make_output_path()
            self._generation.update_progress("inference", 15, 0, 12)
            pipeline_state.pipeline.generate_reference_i2v(
                prompt=enhanced_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                images=anchors,
                video_conditioning=[(control_path, 1.0)],
                lora_path=str(lora_path),
                output_path=str(output_path),
            )
            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise RuntimeError("Generation was cancelled")
            self._generation.update_progress("complete", 100, 12, 12)
            return str(output_path)
        finally:
            for path in prepared_paths:
                if os.path.exists(path):
                    os.unlink(path)
            if control_path and os.path.exists(control_path):
                os.unlink(control_path)

    @staticmethod
    def _reference_frame_indices(reference_count: int, num_frames: int) -> list[int]:
        if reference_count <= 1:
            return [0]
        last = max(num_frames - 1, 0)
        return [
            round(last * index / (reference_count - 1))
            for index in range(reference_count)
        ]

    @staticmethod
    def _reference_control_frame_count(num_frames: int) -> int:
        # The MLX IC-LoRA VAE encoder pads temporal chunks differently from the
        # target latent-shape helper. A control clip with the full target frame
        # count can encode to one extra latent frame (for example 121px -> 17
        # ref latents while the target expects 16). One fewer pixel frame keeps
        # the reference latent sequence aligned to the target grid.
        return max(num_frames - 1, 1)

    def _make_reference_control_video(
        self,
        *,
        prepared_images: list[Image.Image],
        width: int,
        height: int,
        num_frames: int,
        fps: int,
    ) -> str:
        try:
            import cv2
            import imageio_ffmpeg  # type: ignore[reportMissingTypeStubs]
            import numpy as np

            ffmpeg = str(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception as exc:
            raise HTTPError(500, "Local multi-reference generation requires OpenCV and bundled ffmpeg") from exc

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        output_path = tmp.name
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-frames:v",
            str(num_frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            total = max(num_frames - 1, 1)
            for frame_idx in range(num_frames):
                if self._generation.is_generation_cancelled():
                    raise RuntimeError("Generation was cancelled")
                ref_index = round((len(prepared_images) - 1) * frame_idx / total)
                rgb = np.asarray(prepared_images[ref_index].resize((width, height), Image.Resampling.LANCZOS))
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                proc.stdin.write(edge_rgb.tobytes())
                if frame_idx % max(fps, 1) == 0:
                    progress = 14 + int(1 * frame_idx / total)
                    self._generation.update_progress("preparing_references", progress, frame_idx, num_frames)
            proc.stdin.close()
            return_code = proc.wait(timeout=120)
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()

        if return_code != 0:
            Path(output_path).unlink(missing_ok=True)
            raise HTTPError(500, f"Reference control video creation failed: {stderr.strip()}")
        return output_path

    @staticmethod
    def _make_generation_id() -> str:
        return uuid.uuid4().hex[:8]

    @staticmethod
    def _compute_num_frames(duration: int, fps: int) -> int:
        n = ((duration * fps) // 8) * 8 + 1
        return max(n, 9)

    def _resolve_seed(self) -> int:
        settings = self.state.app_settings
        if settings.seed_locked:
            logger.info("Using locked seed: %s", settings.locked_seed)
            return settings.locked_seed
        if self.config.dev_mode:
            return 1000
        return int(time.time()) % 2147483647

    def _make_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.outputs_dir / f"ltx2_video_{timestamp}_{self._make_generation_id()}.mp4"

    def _generate_forced_api(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        generation_id = self._make_generation_id()
        self._generation.start_api_generation(generation_id)

        audio_path = normalize_optional_path(req.audioPath)
        image_path = normalize_optional_path(req.imagePath)
        has_input_audio = bool(audio_path)
        has_input_image = bool(image_path)

        try:
            self._generation.update_progress("validating_request", 5, None, None)

            api_key = self.state.app_settings.ltx_api_key.strip()
            logger.info("Forced API generation route selected (key_present=%s)", bool(api_key))
            if not api_key:
                raise HTTPError(400, "PRO_API_KEY_REQUIRED")

            requested_model = req.model
            api_model_id = FORCED_API_MODEL_MAP.get(requested_model)
            if api_model_id is None:
                raise HTTPError(500, "INVALID_FORCED_API_MODEL_CONFIG")

            resolution_label = req.resolution
            resolution_by_aspect = FORCED_API_RESOLUTION_MAP.get(resolution_label)
            if resolution_by_aspect is None:
                raise HTTPError(500, "INVALID_FORCED_API_RESOLUTION_CONFIG")

            aspect_ratio = req.aspectRatio
            if aspect_ratio not in FORCED_API_ALLOWED_ASPECT_RATIOS:
                raise HTTPError(400, "INVALID_FORCED_API_ASPECT_RATIO")

            api_resolution = resolution_by_aspect[aspect_ratio]

            prompt = req.prompt

            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")

            if has_input_audio:
                validated_audio_path = validate_audio_file(audio_path)
                validated_image_path: Path | None = None
                if image_path is not None:
                    validated_image_path = validate_image_file(image_path)

                self._generation.update_progress("uploading_audio", 20, None, None)
                audio_uri = self._ltx_api_client.upload_file(
                    api_key=api_key,
                    file_path=str(validated_audio_path),
                )
                image_uri: str | None = None
                if validated_image_path is not None:
                    self._generation.update_progress("uploading_image", 35, None, None)
                    image_uri = self._ltx_api_client.upload_file(
                        api_key=api_key,
                        file_path=str(validated_image_path),
                    )
                self._generation.update_progress("inference", 55, None, None)
                video_bytes = self._ltx_api_client.generate_audio_to_video(
                    api_key=api_key,
                    prompt=prompt,
                    audio_uri=audio_uri,
                    image_uri=image_uri,
                    model=api_model_id,
                    resolution=api_resolution,
                )
                self._generation.update_progress("downloading_output", 85, None, None)
            elif has_input_image:
                validated_image_path = validate_image_file(image_path)

                duration = req.duration
                fps = req.fps

                generate_audio = req.audio
                self._generation.update_progress("uploading_image", 20, None, None)
                image_uri = self._ltx_api_client.upload_file(
                    api_key=api_key,
                    file_path=str(validated_image_path),
                )
                self._generation.update_progress("inference", 55, None, None)
                video_bytes = self._ltx_api_client.generate_image_to_video(
                    api_key=api_key,
                    prompt=prompt,
                    image_uri=image_uri,
                    model=api_model_id,
                    resolution=api_resolution,
                    duration=float(duration),
                    fps=float(fps),
                    generate_audio=generate_audio,
                    camera_motion=req.cameraMotion,
                )
                self._generation.update_progress("downloading_output", 85, None, None)
            else:
                duration = req.duration
                fps = req.fps

                generate_audio = req.audio
                self._generation.update_progress("inference", 55, None, None)
                video_bytes = self._ltx_api_client.generate_text_to_video(
                    api_key=api_key,
                    prompt=prompt,
                    model=api_model_id,
                    resolution=api_resolution,
                    duration=float(duration),
                    fps=float(fps),
                    generate_audio=generate_audio,
                    camera_motion=req.cameraMotion,
                )
                self._generation.update_progress("downloading_output", 85, None, None)

            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")

            output_path = self._write_forced_api_video(video_bytes)
            if self._generation.is_generation_cancelled():
                output_path.unlink(missing_ok=True)
                raise RuntimeError("Generation was cancelled")

            self._generation.update_progress("complete", 100, None, None)
            self._generation.complete_generation(str(output_path))
            return GenerateVideoCompleteResponse(status="complete", video_path=str(output_path))
        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            raise
        except LTXAPIClientError as e:
            mapped_error = self._map_ltx_api_generation_error(e)
            self._generation.fail_generation(mapped_error.detail)
            raise mapped_error from e
        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Generation cancelled by user")
                return GenerateVideoCancelledResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e

    def _write_forced_api_video(self, video_bytes: bytes) -> Path:
        output_path = self._make_output_path()
        output_path.write_bytes(video_bytes)
        return output_path

    @staticmethod
    def _map_ltx_api_generation_error(exc: LTXAPIClientError) -> HTTPError:
        if exc.status_code == 402 and exc.provider_error_type == "insufficient_funds_error":
            return HTTPError(402, _LTX_INSUFFICIENT_FUNDS_MESSAGE, code="LTX_INSUFFICIENT_FUNDS")
        return HTTPError(exc.status_code, exc.detail)
