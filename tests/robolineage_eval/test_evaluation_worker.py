from __future__ import annotations

from pathlib import Path

from robolineage_eval.worker import EvaluationReviewWorker


class _FailingEvaluationAgent:
    def run(self, rollout_dir: Path, **_kwargs):
        raise RuntimeError(f"boom:{rollout_dir.name}")


def test_evaluation_worker_allows_retry_after_failure(tmp_path: Path):
    rollout_dir = tmp_path / "rollouts" / "r1"
    rollout_dir.mkdir(parents=True)
    worker = EvaluationReviewWorker(
        agent_factory=lambda: _FailingEvaluationAgent(),
        idle_delay_sec=0.0,
    )
    worker.start()
    try:
        assert worker.enqueue(rollout_dir, policy_version="policy_1") is True
        assert worker.wait_idle(timeout=2.0) is True
        assert "boom:r1" in (worker.status()["last_error"] or "")

        assert worker.enqueue(rollout_dir, policy_version="policy_1") is True
        assert worker.wait_idle(timeout=2.0) is True
    finally:
        worker.stop()
