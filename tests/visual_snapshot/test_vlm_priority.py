from __future__ import annotations

import threading
import time

import numpy as np

from robolineage_shared_agents.visual_snapshot.vlm_priority import (
    OfflineVLMRunner,
    OnlineVLMRunner,
    VLMOnlinePriorityCoordinator,
)
from robolineage_shared_agents.visual_snapshot.vlm_runner import BaseVLMRunner, MockVLMRunner


def test_offline_vlm_waits_while_online_rollout_active():
    coordinator = VLMOnlinePriorityCoordinator()
    coordinator.enter_online("online-1")
    wrapped = MockVLMRunner(fixed_response='{"ok": true}', latency=0.0)
    runner = OfflineVLMRunner(wrapped, coordinator, quiet_period_sec=0.0)
    completed = threading.Event()

    def _run() -> None:
        runner.run("prompt", [np.zeros((4, 4, 3), dtype=np.uint8)])
        completed.set()

    thread = threading.Thread(target=_run)
    thread.start()
    time.sleep(0.05)

    assert not completed.is_set()
    assert coordinator.snapshot().offline_waiting == 1

    coordinator.exit_online("online-1")
    thread.join(timeout=1.0)

    assert completed.is_set()


class _BlockingRunner(BaseVLMRunner):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        self.started.set()
        self.release.wait(timeout=1.0)
        return '{"ok": true}'


class _RecordingRunner(BaseVLMRunner):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, prompt: str, images: list[np.ndarray]) -> str:
        self.calls.append(prompt)
        return '{"ok": true}'


def test_online_rollout_marking_never_waits_for_offline_inflight_call():
    coordinator = VLMOnlinePriorityCoordinator()
    wrapped = _BlockingRunner()
    runner = OfflineVLMRunner(wrapped, coordinator, quiet_period_sec=0.0)

    thread = threading.Thread(
        target=lambda: runner.run("prompt", [np.zeros((4, 4, 3), dtype=np.uint8)])
    )
    thread.start()
    assert wrapped.started.wait(timeout=1.0)

    start = time.perf_counter()
    coordinator.enter_online("online-2")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.02
    snapshot = coordinator.snapshot()
    assert snapshot.online_active is True
    assert snapshot.offline_inflight == 1

    wrapped.release.set()
    thread.join(timeout=1.0)
    coordinator.exit_online("online-2")


def test_online_vlm_runs_linearly_in_rollout_order():
    coordinator = VLMOnlinePriorityCoordinator()
    wrapped = _RecordingRunner()
    coordinator.enter_online("r1")
    coordinator.enter_online("r2")
    r1 = OnlineVLMRunner(wrapped, coordinator, rollout_id="r1")
    r2 = OnlineVLMRunner(wrapped, coordinator, rollout_id="r2")
    r2_done = threading.Event()

    def _run_r2() -> None:
        r2.run("r2-first", [])
        r2_done.set()

    thread = threading.Thread(target=_run_r2)
    thread.start()
    time.sleep(0.05)

    assert not r2_done.is_set()
    assert coordinator.snapshot().online_rollout_queue == ("r1", "r2")

    r1.run("r1-first", [])
    coordinator.exit_online("r1")
    thread.join(timeout=1.0)

    assert r2_done.is_set()
    assert wrapped.calls == ["r1-first", "r2-first"]
    coordinator.exit_online("r2")
