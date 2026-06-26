from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agents import PolicyEvaluationAgent, PolicyEvaluationResult

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationJob:
    rollout_dir: Path
    policy_version: str | None = None
    evaluation_session_id: str | None = None
    evaluation_mode: str = "deployment"


class EvaluationReviewWorker:
    """Single-lane FIFO worker for policy evaluation rollout review."""

    def __init__(
        self,
        *,
        agent_factory: Callable[[], PolicyEvaluationAgent],
        idle_delay_sec: float = 5.0,
    ) -> None:
        self.agent_factory = agent_factory
        self.idle_delay_sec = max(0.0, idle_delay_sec)
        self._queue: queue.Queue[EvaluationJob | object] = queue.Queue()
        self._sentinel = object()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._current_rollout: str | None = None
        self._last_result: PolicyEvaluationResult | None = None
        self._last_error: str | None = None
        self._enqueued: set[Path] = set()
        self._queued_rollouts: list[str] = []

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="PolicyEvaluationReview", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._queue.put(self._sentinel)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                _LOG.warning("evaluation review worker did not stop within %.1fs", timeout)

    def enqueue(
        self,
        rollout_dir: str | Path,
        *,
        policy_version: str | None = None,
        evaluation_session_id: str | None = None,
        evaluation_mode: str = "deployment",
    ) -> bool:
        path = Path(rollout_dir).resolve()
        with self._lock:
            if path in self._enqueued:
                return False
            self._enqueued.add(path)
            self._queued_rollouts.append(path.name)
        self._queue.put(
            EvaluationJob(
                rollout_dir=path,
                policy_version=policy_version,
                evaluation_session_id=evaluation_session_id,
                evaluation_mode=evaluation_mode,
            )
        )
        _LOG.info("queued policy evaluation review for %s", path)
        return True

    def status(self) -> dict:
        with self._lock:
            return {
                "active": self._thread is not None and self._thread.is_alive(),
                "queue_size": self._queue.qsize(),
                "current_rollout": self._current_rollout,
                "queued_rollouts": list(self._queued_rollouts),
                "last_rollout": self._last_result.rollout_id if self._last_result is not None else None,
                "last_error": self._last_error,
            }

    def wait_idle(self, timeout: float | None = 10.0) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._is_idle():
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return self._is_idle()
            time.sleep(0.05)

    def _is_idle(self) -> bool:
        with self._lock:
            return (
                self._queue.qsize() == 0
                and self._current_rollout is None
                and not self._enqueued
            )

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            item = self._queue.get()
            job: EvaluationJob | None = None
            try:
                if item is self._sentinel:
                    return
                assert isinstance(item, EvaluationJob)
                job = item
                with self._lock:
                    self._current_rollout = job.rollout_dir.name
                    if job.rollout_dir.name in self._queued_rollouts:
                        self._queued_rollouts.remove(job.rollout_dir.name)
                    self._last_error = None
                self._sleep_before_review()
                if self._stop_event.is_set():
                    return
                agent = self.agent_factory()
                result = agent.run(
                    job.rollout_dir,
                    policy_version=job.policy_version,
                    evaluation_session_id=job.evaluation_session_id,
                    evaluation_mode=job.evaluation_mode,
                )
                with self._lock:
                    self._last_result = result
                    self._current_rollout = None
            except Exception as exc:
                _LOG.exception("policy evaluation review failed")
                with self._lock:
                    self._last_error = repr(exc)
                    self._current_rollout = None
            finally:
                if job is not None:
                    with self._lock:
                        self._enqueued.discard(job.rollout_dir)
                        if job.rollout_dir.name in self._queued_rollouts:
                            self._queued_rollouts.remove(job.rollout_dir.name)
                self._queue.task_done()

    def _sleep_before_review(self) -> None:
        deadline = time.monotonic() + self.idle_delay_sec
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(min(0.1, deadline - time.monotonic()))
