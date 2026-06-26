from __future__ import annotations

from pathlib import Path

from robolineage_post_rollout.worker import PostRolloutReviewWorker


class _FailingReviewAgent:
    def run(self, rollout_dir: Path):
        raise RuntimeError(f"boom:{rollout_dir.name}")


def test_post_rollout_worker_allows_retry_after_failure(tmp_path: Path):
    rollout_dir = tmp_path / "rollouts" / "r1"
    rollout_dir.mkdir(parents=True)
    worker = PostRolloutReviewWorker(
        agent_factory=lambda: _FailingReviewAgent(),
        idle_delay_sec=0.0,
    )
    worker.start()
    try:
        assert worker.enqueue(rollout_dir) is True
        assert worker.wait_idle(timeout=2.0) is True
        assert "boom:r1" in (worker.status()["last_error"] or "")

        assert worker.enqueue(rollout_dir) is True
        assert worker.wait_idle(timeout=2.0) is True
    finally:
        worker.stop()
