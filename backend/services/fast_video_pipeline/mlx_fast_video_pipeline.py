"""Apple Silicon MLX fast video pipeline wrapper."""

from __future__ import annotations

import json
import logging
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar, Literal, NoReturn

from api_types import ImageConditioningInput

logger = logging.getLogger(__name__)


class MLXFastVideoPipeline:
    pipeline_kind: ClassVar[Literal["fast"]] = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: object,
        streaming_prefetch_count: int | None,
    ) -> "MLXFastVideoPipeline":
        del upsampler_path, device, streaming_prefetch_count
        return MLXFastVideoPipeline(
            model_dir=Path(checkpoint_path),
            gemma_dir=Path(gemma_root) if gemma_root else None,
        )

    def __init__(self, *, model_dir: Path, gemma_dir: Path | None) -> None:
        self._model_dir = model_dir
        self._gemma_dir = gemma_dir
        self._proc: subprocess.Popen[str] | None = None
        self._ready_info: dict[str, Any] = {}

    def _resolve_mlx_root(self) -> Path:
        env_root = os.environ.get("LTX_MLX_PATH")
        candidates: list[Path] = []
        if env_root:
            candidates.append(Path(env_root))
        candidates.extend(
            [
                self._model_dir.parent.parent / "ltx-2-mlx",
                self._model_dir.parent / "ltx-2-mlx",
            ]
        )
        for candidate in candidates:
            if (candidate / "packages").exists() or (candidate / "pyproject.toml").exists():
                return candidate
        raise RuntimeError(
            "LTX_MLX_PATH is not configured and no ltx-2-mlx checkout was found near the models directory. "
            "Run scripts/setup-mac-mlx.sh or set LTX_MLX_PATH to a prepared ltx-2-mlx checkout."
        )

    def _resolve_helper_python(self, mlx_root: Path) -> Path:
        env_python = os.environ.get("LTX_MLX_PYTHON")
        candidates: list[Path] = []
        if env_python:
            candidates.append(Path(env_python))
        candidates.extend(
            [
                mlx_root / "env" / "bin" / "python3.11",
                mlx_root / ".venv" / "bin" / "python3.11",
                Path(sys.executable),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise RuntimeError(
            "Could not find a Python 3.11 environment for ltx-2-mlx. "
            "Run scripts/setup-mac-mlx.sh before local Mac generation."
        )

    def _helper_script(self) -> Path:
        return Path(__file__).with_name("mlx_warm_helper.py")

    def _ffmpeg_bin_dir(self) -> Path | None:
        try:
            import imageio_ffmpeg  # type: ignore[reportMissingTypeStubs]

            ffmpeg_exe = Path(str(imageio_ffmpeg.get_ffmpeg_exe()))
        except Exception:
            return None
        if not ffmpeg_exe.exists():
            return None
        if ffmpeg_exe.name == "ffmpeg":
            return ffmpeg_exe.parent

        shim_dir = Path(tempfile.gettempdir()) / "ltx-desktop-ffmpeg"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_path = shim_dir / "ffmpeg"
        if not shim_path.exists():
            try:
                shim_path.symlink_to(ffmpeg_exe)
            except FileExistsError:
                pass
        return shim_dir

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        if not self._model_dir.exists():
            raise RuntimeError(f"MLX model directory not found: {self._model_dir}")
        if self._gemma_dir is None or not self._gemma_dir.exists():
            raise RuntimeError(
                "MLX Gemma text encoder is not downloaded. Download gemma-3-12b-it-4bit or enter an API key only for prompt enhancement."
            )

        mlx_root = self._resolve_mlx_root()
        helper_python = self._resolve_helper_python(mlx_root)
        helper_script = self._helper_script()

        env = os.environ.copy()
        env["LTX_MODEL"] = str(self._model_dir)
        env["LTX_GEMMA"] = str(self._gemma_dir)
        env.setdefault("LTX_LOW_MEMORY", "true")
        env.setdefault("LTX_IDLE_TIMEOUT", "1800")
        env.setdefault("LTX_ENABLE_MODEL_UPSCALE", "0")
        package_paths = [
            mlx_root / "packages" / "ltx-core-mlx" / "src",
            mlx_root / "packages" / "ltx-pipelines-mlx" / "src",
            mlx_root / "packages" / "ltx-trainer" / "src",
        ]
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(path) for path in package_paths if path.exists()]
            + ([existing_pythonpath] if existing_pythonpath else [])
        )
        ffmpeg_bin_dir = self._ffmpeg_bin_dir()
        if ffmpeg_bin_dir is not None:
            env["PATH"] = f"{ffmpeg_bin_dir}:{env.get('PATH', '')}"

        self._proc = subprocess.Popen(
            [str(helper_python), str(helper_script)],
            cwd=str(mlx_root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        ready = self._read_until({"ready", "error", "exit"}, timeout=180)
        if ready is None or ready.get("event") != "ready":
            self._kill()
            raise RuntimeError(f"MLX helper failed to start: {ready}")
        self._ready_info = ready
        logger.info("MLX helper ready: %s", {k: ready.get(k) for k in ("model", "gemma", "low_memory", "ltx_version")})

    def _read_until(self, target_events: set[str], timeout: float | None = None) -> dict[str, Any] | None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        deadline = time.time() + timeout if timeout is not None else None
        while True:
            if deadline is not None and time.time() > deadline:
                return None
            wait = 0.5
            rlist, _, _ = select.select([proc.stdout], [], [], wait)
            if not rlist:
                continue
            line = proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.info("MLX helper: %s", line)
                continue

            event_type = event.get("event")
            if event_type == "log":
                logger.info("MLX helper: %s", event.get("line", ""))
                continue
            if event_type in target_events:
                return event
            logger.info("MLX helper event: %s", event)

    def _read_job_until_done(self, *, job_id: str, output_path: str) -> dict[str, Any] | None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None

        decode_done_at: float | None = None
        grace_seconds = 45.0
        min_file_bytes = 8 * 1024
        while True:
            if decode_done_at is not None and (time.time() - decode_done_at) >= grace_seconds:
                output = Path(output_path)
                if output.exists() and output.stat().st_size >= min_file_bytes:
                    logger.warning(
                        "MLX helper appears hung after decode; killing helper and accepting completed file: %s",
                        output,
                    )
                    self._kill()
                    return {
                        "event": "done",
                        "id": job_id,
                        "output": str(output),
                        "watchdog_forced_exit": True,
                    }

            rlist, _, _ = select.select([proc.stdout], [], [], 0.5)
            if not rlist:
                continue
            line = proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.info("MLX helper: %s", line)
                continue

            event_type = event.get("event")
            if event_type == "log":
                log_line = str(event.get("line", ""))
                logger.info("MLX helper: %s", log_line)
                if "[Decoding video + audio + muxing] done in" in log_line:
                    decode_done_at = time.time()
                continue
            if event_type in {"done", "error", "exit"}:
                return event
            logger.info("MLX helper event: %s", event)

    def _run(self, job: dict[str, Any]) -> dict[str, Any]:
        self._ensure()
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("MLX helper is not running")
        try:
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._raise_for_helper_exit(prefix=f"MLX helper stdin closed: {exc}")

        params = job.get("params", {})
        output_path = str(params.get("output_path", ""))
        job_id = str(job.get("id", "?"))
        event = self._read_job_until_done(job_id=job_id, output_path=output_path)
        if event is None:
            self._raise_for_helper_exit(prefix="MLX helper pipe closed without a completion event")
        if event.get("event") == "error":
            message = str(event.get("error", "MLX helper error"))
            trace = event.get("trace")
            if trace:
                message = f"{message}\n{trace}"
            raise RuntimeError(message)
        if event.get("event") == "exit":
            raise RuntimeError(f"MLX helper exited: {event.get('reason')}")
        return event

    def _raise_for_helper_exit(self, *, prefix: str) -> NoReturn:
        proc = self._proc
        rc = proc.poll() if proc is not None else None
        if rc is not None and rc < 0:
            signal_name = signal.Signals(-rc).name
            raise RuntimeError(f"{prefix}; helper exited from {signal_name}. On a 36GB Mac this usually means the render exceeded memory headroom.")
        raise RuntimeError(f"{prefix}; returncode={rc}")

    def _kill(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

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
        image_path = images[0].path if images else None
        accel = "off" if speed_mode == "quality" else speed_mode
        job: dict[str, Any] = {
            "action": "generate",
            "id": uuid.uuid4().hex[:8],
            "params": {
                "mode": "i2v" if image_path else "t2v",
                "prompt": prompt,
                "negative_prompt": "",
                "output_path": output_path,
                "height": int(height),
                "width": int(width),
                "frames": int(num_frames),
                "frame_rate": float(frame_rate),
                "seed": int(seed),
                "steps": 8,
                "accel": accel,
                "upscale": "off",
                "upscale_method": "lanczos",
                "loras": [],
                "image": image_path,
            },
        }
        result = self._run(job)
        output = result.get("output")
        if output is None or not Path(str(output)).exists():
            raise RuntimeError(f"MLX helper completed but output was not found: {output}")

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
        job: dict[str, Any] = {
            "action": "generate_ic_lora",
            "id": uuid.uuid4().hex[:8],
            "params": {
                "prompt": prompt,
                "output_path": output_path,
                "height": int(height),
                "width": int(width),
                "frames": int(num_frames),
                "frame_rate": float(frame_rate),
                "seed": int(seed),
                "stage1_steps": 8,
                "stage2_steps": 0,
                "loras": [{"path": lora_path, "strength": 1.0}],
                "video_conditioning": video_conditioning,
                # The control video already carries the local multi-image
                # references. Direct MLX image anchors currently hit a reshape
                # bug in combined_image_conditionings for this Q4 IC-LoRA path.
                "images": [],
                "conditioning_attention_strength": 1.0,
                "skip_stage_2": True,
            },
        }
        result = self._run(job)
        output = result.get("output")
        if output is None or not Path(str(output)).exists():
            raise RuntimeError(f"MLX helper completed but output was not found: {output}")

    def enhance_prompt(self, prompt: str, *, mode: Literal["t2v", "i2v"], seed: int) -> str:
        job: dict[str, Any] = {
            "action": "enhance_prompt",
            "id": uuid.uuid4().hex[:8],
            "params": {
                "prompt": prompt,
                "mode": mode,
                "seed": int(seed),
                "preserve_tokens": [],
            },
        }
        result = self._run(job)
        enhanced = result.get("enhanced")
        if not isinstance(enhanced, str) or not enhanced.strip():
            raise RuntimeError("MLX helper prompt enhancement completed without an enhanced prompt")
        return enhanced.strip()

    def warmup(self, output_path: str) -> None:
        self.generate(
            prompt="test warmup",
            seed=42,
            height=256,
            width=384,
            num_frames=9,
            frame_rate=8,
            images=[],
            output_path=output_path,
        )
        Path(output_path).unlink(missing_ok=True)

    def compile_transformer(self) -> None:
        logger.info("Skipping torch.compile() for MLX pipeline")
