from __future__ import annotations

import json
from pathlib import Path

from robolineage_post_rollout.regression_metrics import summarize_post_review_regression


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_post_review_regression_metrics_summarizes_rollout_artifacts(tmp_path: Path):
    rollouts_dir = tmp_path / "rollouts"
    first = rollouts_dir / "rollout_a"
    second = rollouts_dir / "rollout_b"

    _write_json(
        first / "evidence_index.json",
        {
            "image_count": 4,
            "final_observation_frames": [20],
            "vsa_windows": [
                {"event_type": "final_observation", "anchor_frame_id": 20},
                {"event_type": "periodic_sample", "anchor_frame_id": 10},
                {"event_type": "periodic_sample", "anchor_frame_id": 10},
            ],
        },
    )
    _write_json(
        first / "annotation.final.json",
        {
            "outcome": {"final_success": True},
            "phase_timeline": [
                {"phase": "approach", "duration_sec": 0.0},
                {"phase": "finish", "duration_sec": 1.2},
            ],
        },
    )
    _write_json(first / "rollout_summary.json", {"final_success": True})
    _write_json(
        first / "dataset_admission.json",
        {
            "decision": "needs_review",
            "accepted_for_training": True,
            "label_quality": "uncertain",
        },
    )

    _write_json(second / "evidence_index.json", {"image_count": 1, "vsa_windows": []})
    _write_json(
        second / "annotation.final.json",
        {"outcome": {"final_success": False}, "phase_timeline": []},
    )
    _write_json(second / "rollout_summary.json", {"final_success": True})
    _write_json(second / "dataset_admission.json", {"decision": "rejected", "accepted_for_training": False})

    summary = summarize_post_review_regression(rollouts_dir)

    assert summary["rollout_count"] == 2
    assert summary["metrics"]["image_count_total"] == 5
    assert summary["metrics"]["final_observation_frame_count"] == 1
    assert summary["metrics"]["duplicate_window_count"] == 1
    assert summary["metrics"]["zero_duration_phase_count"] == 1
    assert summary["metrics"]["accepted_for_training_count"] == 1
    assert summary["metrics"]["final_success_compared_count"] == 2
    assert summary["metrics"]["final_success_aligned_count"] == 1
    assert summary["metrics"]["final_success_alignment_rate"] == 0.5
    assert summary["rollouts"][0]["rollout_id"] == "rollout_a"
    assert summary["rollouts"][0]["duplicate_window_count"] == 1
