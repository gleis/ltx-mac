from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Literal, TypeVar

from services.store import Store
from services.records import (
    AttemptError,
    CapabilityFailedError,
    GenerationErrorCode,
    GenerationRecord,
    MarkSucceededError,
    OutputSpec,
    StatusError,
    UnavailableError,
)
from _routes._errors import HTTPError
from logging_policy import log_background_exception
from runtime_config.reject_reasons import BUDGET_MISS_CODES
from services.analytics import (
    AnalyticsService,
    QueuedEnhancement,
    QueuedExecutionModes,
    QueuedHardwareSnapshot,
    queued_generation_details,
)
from services.generation_queue.control import (
    GenerationSlotWaitAborted,
    SharedGenerationSlot,
)
from services.generation_queue.output_names import output_asset_name
from services.generation_queue.registry import ExecutorRegistry
from services.generation_queue.types import ExecutionOutcome, OutputAllocation
from services.generation_interrupt import is_cancel_exception
from services.gpu_info.gpu_info import GpuInfo
from services.patches import diffusion_stage_cache
from services.patches.pinned_pool_fix import is_cuda_oom_error
from services.storage_info.storage_info import DriveType

logger = logging.getLogger(__name__)
_BACKGROUND_TASK_NAME = "generation-queue-runner"
_EXECUTOR_FAILURE_CODE: GenerationErrorCode = "EXECUTOR_FAILED"
_OOM_CODE: GenerationErrorCode = "CUDA_OOM"
_CAPABILITY_FAILURE_CODE: GenerationErrorCode = "CAPABILITY_FAILED"
_JOB_TOO_LARGE_CODE: GenerationErrorCode = "JOB_TOO_LARGE"
# BUDGET_MISS_CODES (imported above) are budget verdicts, not true failures:
# an executor raising one means the job stopped fitting between create and
# execute. Mapped to JOB_TOO_LARGE with the enriched detail preserved in
# the log; anything else keeps the branch default.
_LTX_INVALID_API_KEY_CODE: GenerationErrorCode = "LTX_INVALID_API_KEY"
_LTX_API_PROMPT_EMBEDDING_FAILED_CODE: GenerationErrorCode = "LTX_API_PROMPT_EMBEDDING_FAILED"


def _error_code_for(exc: BaseException, default: GenerationErrorCode) -> GenerationErrorCode:
    """OOM takes precedence over the branch default.

    Edit executors re-raise torch OOMs (a RuntimeError subclass) as
    capability failures, so the capability branch must check too — otherwise
    the policy-miss signal hides as a generic failure.
    """
    if is_cuda_oom_error(exc):
        return _OOM_CODE
    return default


_T = TypeVar("_T")


def _probe_or_none(probe: Callable[[], _T | None]) -> _T | None:
    """One tolerant probe: exceptions and missing values both become None."""
    try:
        return probe()
    except Exception:
        return None


def _capability_error_code(exc: CapabilityFailedError) -> GenerationErrorCode:
    if exc.code == _LTX_INVALID_API_KEY_CODE:
        return _LTX_INVALID_API_KEY_CODE
    if exc.code == _LTX_API_PROMPT_EMBEDDING_FAILED_CODE:
        return _LTX_API_PROMPT_EMBEDDING_FAILED_CODE
    if exc.code in BUDGET_MISS_CODES:
        return _JOB_TOO_LARGE_CODE
    return _CAPABILITY_FAILURE_CODE


def _http_budget_error_code(exc: HTTPError) -> GenerationErrorCode:
    """Budget-verdict HTTP codes map to JOB_TOO_LARGE; the rest are true failures."""
    if exc.code in BUDGET_MISS_CODES:
        return _JOB_TOO_LARGE_CODE
    return _EXECUTOR_FAILURE_CODE


@dataclass
class _ActiveGeneration:
    generation_id: str
    attempt_count: int


class QueueRunner:
    def __init__(
        self,
        *,
        db: Store,
        generation: SharedGenerationSlot,
        executors: ExecutorRegistry,
        analytics: AnalyticsService | None = None,
        enhancement_policy: Callable[[GenerationRecord], QueuedEnhancement] | None = None,
        gpu_info: GpuInfo | None = None,
        checkpoint_drive_type: DriveType | None = None,
        idle_wait_seconds: float = 0.25,
        stop_join_timeout_seconds: float = 600.0,
    ) -> None:
        self._db = db
        self._generation = generation
        self._executors = executors
        self._analytics = analytics
        self._enhancement_policy = enhancement_policy
        self._gpu_info = gpu_info
        self._checkpoint_drive_type: DriveType | None = checkpoint_drive_type
        self._idle_wait_seconds = idle_wait_seconds
        self._stop_join_timeout_seconds = stop_join_timeout_seconds

        self._shutdown = Event()
        self._wake_event = Event()
        self._lifecycle_lock = Lock()
        self._thread: Thread | None = None
        self._active: _ActiveGeneration | None = None
        self._finished_listener: Callable[[str], None] | None = None

    def set_finished_listener(self, listener: Callable[[str], None] | None) -> None:
        self._finished_listener = listener

    def set_checkpoint_drive_type(self, drive_type: DriveType | None) -> None:
        """Refresh the telemetry snapshot's drive class (models dir moved)."""
        self._checkpoint_drive_type = drive_type

    def start(self) -> None:
        try:
            self._db.has_queued()
        except UnavailableError:
            return
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown.clear()
            self._wake_event.clear()
            self._thread = Thread(
                target=self._run,
                name="generation-queue-runner",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._shutdown.set()
            self._wake_event.set()

        if thread is not current_thread():
            thread.join(timeout=self._stop_join_timeout_seconds)

        with self._lifecycle_lock:
            if self._thread is thread and not thread.is_alive():
                self._thread = None

    def wake(self) -> None:
        self._wake_event.set()

    def interrupt(self, generation_id: str, attempt_count: int) -> None:
        with self._lifecycle_lock:
            active = self._active
            if (
                active is None
                or active.generation_id != generation_id
                or active.attempt_count != attempt_count
            ):
                return
            try:
                self._generation.cancel_generation()
            except Exception as exc:
                logger.warning(
                    "Generation %s attempt %s interrupt failed: %s",
                    generation_id,
                    attempt_count,
                    exc,
                )

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while True:
            if self._shutdown.is_set():
                return
            try:
                if not self._db.has_queued():
                    self._wake_event.wait(timeout=self._idle_wait_seconds)
                    self._wake_event.clear()
                    continue

                with self._generation.wait_for_generation_slot(self._shutdown):
                    claimed = self._claim_and_register_active()
                    if claimed is None:
                        if self._shutdown.is_set():
                            return
                        continue
                    try:
                        self._process(claimed)
                    finally:
                        self._clear_active(claimed)
            except GenerationSlotWaitAborted:
                return
            except Exception as exc:
                log_background_exception(_BACKGROUND_TASK_NAME, exc)
                if self._shutdown.wait(timeout=self._idle_wait_seconds):
                    return

    def _claim_and_register_active(self) -> GenerationRecord | None:
        if self._shutdown.is_set():
            return None
        claimed = self._db.claim_next_queued()
        if claimed is None:
            return None
        with self._lifecycle_lock:
            self._active = _ActiveGeneration(
                generation_id=claimed.id,
                attempt_count=claimed.attempt_count,
            )
        try:
            latest = self._db.get_generation(claimed.id)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            self._clear_active(claimed)
            return None
        if latest is None or latest.attempt_count != claimed.attempt_count:
            self._clear_active(claimed)
            return None
        if latest.status == "cancelling":
            self._send_generation_started(latest)
            self._finalize_cancellation_or_unlink(claimed, [])
            self._clear_active(claimed)
            return None
        if latest.status != "running":
            self._clear_active(claimed)
            return None
        self._send_generation_started(latest)
        return claimed

    def _process(self, claimed: GenerationRecord) -> None:
        allocations: list[OutputAllocation] = []
        partial_paths: list[str] = []
        try:
            executor = self._executors.get(claimed.feature)
            plans = executor.plan_outputs(claimed)
            for plan in plans:
                asset_id, dest_path = self._db.allocate_output_path(
                    plan.media_kind, plan.mime_type
                )
                allocations.append(
                    OutputAllocation(
                        plan=plan,
                        asset_id=asset_id,
                        dest_path=dest_path,
                    )
                )
                partial_paths.append(dest_path)
            # Release dead pools (cached transformer + allocator reserved)
            # before the execute-time budget samples free: start_generation
            # evicts after the sample, so sampling first charges the job
            # for D it never sees. Warm live weights W still count against the
            # first sample; asks landing in (free', free'+W] get one
            # evict-and-resample inside the budget gates themselves (see
            # runtime_config.evict_retry) instead of failing outright.
            # Best-effort: a reclaim failure must never fail the job it was
            # trying to save.
            try:
                diffusion_stage_cache.evict()
            except Exception:
                logger.warning(
                    "Generation %s reclaim before execute failed; continuing",
                    claimed.id,
                    exc_info=True,
                )
            outcome = executor.execute(claimed, tuple(allocations))
            allocations = [
                allocation
                for allocation in allocations
                if not allocation.plan.optional or Path(allocation.dest_path).is_file()
            ]
        except Exception as exc:
            if is_cancel_exception(exc):
                self._mark_cancelled(claimed, partial_paths)
            elif isinstance(exc, CapabilityFailedError):
                logger.warning(
                    "Generation %s attempt %s capability error: %s",
                    claimed.id,
                    claimed.attempt_count,
                    exc.detail,
                )
                self._mark_failed(
                    claimed, _error_code_for(exc, _capability_error_code(exc)), partial_paths
                )
            elif isinstance(exc, HTTPError):
                logger.warning(
                    "Generation %s attempt %s HTTP error at execute time: %s",
                    claimed.id,
                    claimed.attempt_count,
                    exc.detail,
                )
                self._mark_failed(
                    claimed, _error_code_for(exc, _http_budget_error_code(exc)), partial_paths
                )
            else:
                log_background_exception(_BACKGROUND_TASK_NAME, exc)
                self._mark_failed(claimed, _error_code_for(exc, _EXECUTOR_FAILURE_CODE), partial_paths)
            return

        self._mark_succeeded(claimed, allocations, outcome)

    def _mark_succeeded(
        self,
        claimed: GenerationRecord,
        allocations: list[OutputAllocation],
        outcome: ExecutionOutcome | None,
    ) -> None:
        params = claimed.spec.get("params")
        prompt: object = None
        if isinstance(params, dict):
            prompt = params.get("prompt")
        written_prompt = outcome.written_prompt if outcome is not None else None
        if written_prompt is not None:
            prompt = written_prompt
        outputs = [
            OutputSpec(
                asset_id=allocation.asset_id,
                dest_path=allocation.dest_path,
                ordinal=index,
                mime_type=allocation.plan.mime_type,
                name=output_asset_name(
                    prompt=prompt,
                    created_at_ms=claimed.created_at,
                    asset_id=allocation.asset_id,
                    mime_type=allocation.plan.mime_type,
                ),
            )
            for index, allocation in enumerate(allocations)
        ]
        paths = [output.dest_path for output in outputs]
        if self._is_cancelling_attempt(claimed):
            self._finalize_cancellation_or_unlink(claimed, paths)
            return
        try:
            completed = self._db.mark_succeeded(
                claimed.id,
                outputs,
                attempt_count=claimed.attempt_count,
                written_prompt=written_prompt,
            )
        except MarkSucceededError as exc:
            self._mark_failed(claimed, exc.error_code, paths)
        except (StatusError, AttemptError):
            self._finalize_cancellation_or_unlink(claimed, paths)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            self._unlink_unless_succeeded(claimed.id, paths)
        else:
            self._notify_finished(claimed.id)
            self._send_generation_ended(completed, outcome)

    def _is_cancelling_attempt(self, claimed: GenerationRecord) -> bool:
        try:
            latest = self._db.get_generation(claimed.id)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            return False
        return (
            latest is not None
            and latest.status == "cancelling"
            and latest.attempt_count == claimed.attempt_count
        )

    def _mark_failed(
        self,
        claimed: GenerationRecord,
        error_code: GenerationErrorCode,
        partial_paths: list[str],
    ) -> None:
        wrote = self._cas_or_unlink(
            claimed,
            partial_paths,
            lambda: self._db.mark_failed(
                claimed.id,
                error_code,
                partial_paths,
                attempt_count=claimed.attempt_count,
            ),
        )
        if wrote:
            self._notify_finished(claimed.id)

    def _notify_finished(self, generation_id: str) -> None:
        listener = self._finished_listener
        if listener is None:
            return
        try:
            listener(generation_id)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)

    def _mark_cancelled(
        self,
        claimed: GenerationRecord,
        partial_paths: list[str],
    ) -> None:
        self._cas_or_unlink(
            claimed,
            partial_paths,
            lambda: self._db.mark_cancelled(
                claimed.id,
                partial_paths,
                attempt_count=claimed.attempt_count,
            ),
        )

    def _cas_or_unlink(
        self,
        claimed: GenerationRecord,
        paths: list[str],
        write: Callable[[], object],
    ) -> bool:
        try:
            completed = write()
        except (StatusError, AttemptError):
            self._finalize_cancellation_or_unlink(claimed, paths)
            return False
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            self._unlink_unless_succeeded(claimed.id, paths)
            return False
        if isinstance(completed, GenerationRecord):
            self._send_generation_ended(completed)
        return True

    def _finalize_cancellation_or_unlink(
        self,
        claimed: GenerationRecord,
        paths: list[str],
    ) -> None:
        try:
            completed = self._db.mark_cancelled(
                claimed.id,
                paths,
                attempt_count=claimed.attempt_count,
            )
        except (StatusError, AttemptError):
            self._unlink_unless_succeeded(claimed.id, paths)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            self._unlink_unless_succeeded(claimed.id, paths)
        else:
            self._send_generation_ended(completed)

    def _unlink_unless_succeeded(self, generation_id: str, paths: list[str]) -> None:
        try:
            generation = self._db.get_generation(generation_id)
        except Exception as exc:
            logger.warning(
                "Could not inspect generation %s after a lost finalization; "
                "preserving allocated paths: %s",
                generation_id,
                exc,
            )
            return
        if generation is None or generation.status != "succeeded":
            self._db.unlink_partial_outputs(paths)

    def _enhancement(self, generation: GenerationRecord) -> QueuedEnhancement | None:
        if self._enhancement_policy is None:
            return None
        return self._enhancement_policy(generation)

    def _snapshot_hardware(self) -> QueuedHardwareSnapshot | None:
        """Best-effort hardware snapshot for the started event.

        Per-probe tolerance: each field degrades to None independently, and
        anything unexpected degrades the whole snapshot. Telemetry never
        breaks generation.
        """
        if self._gpu_info is None:
            return None
        try:
            return QueuedHardwareSnapshot(
                gpu_model=_probe_or_none(self._gpu_info.get_device_name),
                total_vram_bytes=_probe_or_none(self._gpu_info.get_cuda_total_vram_bytes),
                free_vram_bytes=_probe_or_none(self._gpu_info.get_free_cuda_vram_bytes),
                drive_type=self._checkpoint_drive_type,
            )
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            return None

    def _send_generation_started(self, generation: GenerationRecord) -> None:
        if self._analytics is None:
            return
        try:
            details = queued_generation_details(
                generation,
                enhancement=self._enhancement(generation),
                hardware=self._snapshot_hardware(),
            )
            if details is not None:
                self._analytics.send_generation_started(details)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)

    def _send_generation_ended(
        self, generation: GenerationRecord, summary: ExecutionOutcome | None = None
    ) -> None:
        if (
            self._analytics is None
            or generation.started_at is None
            or generation.finished_at is None
        ):
            return
        try:
            modes: QueuedExecutionModes | None = (
                {"load_mode": summary.load_mode, "offload_mode": summary.offload_mode}
                if summary is not None
                else None
            )
            details = queued_generation_details(
                generation, enhancement=self._enhancement(generation), modes=modes
            )
            if details is None:
                return
            outcome: Literal["succeeded", "failed", "cancelled"]
            if generation.status == "succeeded":
                outcome = "succeeded"
            elif generation.status == "failed":
                outcome = "failed"
            elif generation.status == "cancelled":
                outcome = "cancelled"
            else:
                return
            self._analytics.send_generation_ended(
                details,
                outcome=outcome,
                runtime_ms=generation.finished_at - generation.started_at,
                error_code=generation.error_code,
            )
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)

    def _clear_active(self, generation: GenerationRecord) -> None:
        with self._lifecycle_lock:
            if (
                self._active is not None
                and self._active.generation_id == generation.id
                and self._active.attempt_count == generation.attempt_count
            ):
                self._active = None
