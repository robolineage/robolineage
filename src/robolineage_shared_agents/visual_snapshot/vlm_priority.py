from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .exceptions import VLMInferenceError
from .vlm_runner import BaseVLMRunner


@dataclass(frozen=True)
class VLMUsageSnapshot:
    online_active: bool
    online_rollout_id: str | None
    online_rollout_queue: tuple[str, ...]
    offline_waiting: int
    offline_inflight: int
    last_online_ended_at: float | None


class VLMOnlinePriorityCoordinator:
    """Coordinates online and offline VLM lanes.

    Online VSA never waits on this coordinator. It only marks rollout
    activity. Offline review calls must acquire an offline slot before using
    VLM; they block while an online rollout is active and observe a short quiet
    period after online inference stops.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._online_rollout_order: list[str] = []
        self._online_inflight = False
        self._offline_waiting = 0
        self._offline_inflight = 0
        self._last_online_ended_at: float | None = None

    def enter_online(self, rollout_id: str | None = None) -> None:
        with self._condition:
            if rollout_id is not None and rollout_id not in self._online_rollout_order:
                self._online_rollout_order.append(rollout_id)
            self._condition.notify_all()

    def exit_online(self, rollout_id: str | None = None) -> None:
        with self._condition:
            if rollout_id is None:
                self._online_rollout_order.clear()
            else:
                self._online_rollout_order = [
                    item for item in self._online_rollout_order if item != rollout_id
                ]
            if not self._online_rollout_order:
                self._last_online_ended_at = time.monotonic()
            self._condition.notify_all()

    @contextmanager
    def online_slot(
        self,
        rollout_id: str,
        *,
        stop_event: threading.Event | None = None,
        poll_interval_sec: float = 0.1,
    ) -> Iterator[None]:
        """Serialize online VLM calls in rollout order.

        Multiple rollout pipelines may be draining while the operator has
        already started the next rollout. This slot keeps the actual VLM
        backend strictly linear and prevents later rollouts from overtaking
        earlier queued online analysis.
        """
        acquired = False
        with self._condition:
            if rollout_id not in self._online_rollout_order:
                self._online_rollout_order.append(rollout_id)
            while True:
                if stop_event is not None and stop_event.is_set():
                    raise VLMInferenceError("online VLM stopped before inference")
                is_turn = self._online_rollout_order and self._online_rollout_order[0] == rollout_id
                if is_turn and not self._online_inflight:
                    self._online_inflight = True
                    acquired = True
                    break
                self._condition.wait(timeout=poll_interval_sec)
        try:
            yield
        finally:
            if acquired:
                with self._condition:
                    self._online_inflight = False
                    self._condition.notify_all()

    @contextmanager
    def offline_slot(
        self,
        *,
        stop_event: threading.Event | None = None,
        quiet_period_sec: float = 0.0,
        poll_interval_sec: float = 0.1,
    ) -> Iterator[None]:
        acquired = False
        with self._condition:
            self._offline_waiting += 1
            try:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        raise VLMInferenceError("offline VLM stopped before inference")
                    wait_for = self._wait_time_locked(quiet_period_sec)
                    if wait_for <= 0:
                        break
                    self._condition.wait(timeout=min(wait_for, poll_interval_sec))
                self._offline_inflight += 1
                acquired = True
            finally:
                self._offline_waiting -= 1
        try:
            yield
        finally:
            if acquired:
                with self._condition:
                    self._offline_inflight -= 1
                    self._condition.notify_all()

    def _wait_time_locked(self, quiet_period_sec: float) -> float:
        if self._online_rollout_order or self._online_inflight:
            return 3600.0
        if quiet_period_sec <= 0 or self._last_online_ended_at is None:
            return 0.0
        elapsed = time.monotonic() - self._last_online_ended_at
        return max(0.0, quiet_period_sec - elapsed)

    def snapshot(self) -> VLMUsageSnapshot:
        with self._condition:
            return VLMUsageSnapshot(
                online_active=bool(self._online_rollout_order) or self._online_inflight,
                online_rollout_id=(
                    self._online_rollout_order[0]
                    if self._online_rollout_order
                    else None
                ),
                online_rollout_queue=tuple(self._online_rollout_order),
                offline_waiting=self._offline_waiting,
                offline_inflight=self._offline_inflight,
                last_online_ended_at=self._last_online_ended_at,
            )


class OnlineVLMRunner(BaseVLMRunner):
    """VLM runner wrapper for online VSA.

    It does not change prompts, images, parsing, or phase logic. It only gates
    access to the underlying backend so online VSA calls remain globally
    linear even when multiple stopped rollouts are draining in the background.
    """

    def __init__(
        self,
        runner: BaseVLMRunner,
        coordinator: VLMOnlinePriorityCoordinator,
        *,
        rollout_id: str,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.runner = runner
        self.coordinator = coordinator
        self.rollout_id = rollout_id
        self.stop_event = stop_event

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        with self.coordinator.online_slot(
            self.rollout_id,
            stop_event=self.stop_event,
        ):
            return self.runner.run(prompt, images)


class OfflineVLMRunner(BaseVLMRunner):
    """VLM runner wrapper for post-rollout review.

    It shares the same underlying API/backend configuration as online VSA, but
    never shares a realtime queue. Each call pauses before reaching the backend
    whenever an online rollout is active.
    """

    def __init__(
        self,
        runner: BaseVLMRunner,
        coordinator: VLMOnlinePriorityCoordinator,
        *,
        stop_event: threading.Event | None = None,
        quiet_period_sec: float = 5.0,
    ) -> None:
        self.runner = runner
        self.coordinator = coordinator
        self.stop_event = stop_event
        self.quiet_period_sec = max(0.0, quiet_period_sec)

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        with self.coordinator.offline_slot(
            stop_event=self.stop_event,
            quiet_period_sec=self.quiet_period_sec,
        ):
            return self.runner.run(prompt, images)
