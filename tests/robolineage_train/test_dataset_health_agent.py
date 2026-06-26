from __future__ import annotations

import json
from pathlib import Path

from robolineage_train.dataset_health import DatasetHealthAgent


class _FakeDatasetHealthLLM:
    model = "fake-dataset-health-llm"

    def generate(self, context: dict) -> dict:
        assert context["deterministic_report"]["selected_rollout_count"] == 2
        assert context["deterministic_report"]["phase_coverage"]["missing_phases"] == ["place"]
        return {
            "summary": "Dataset is useful but under-covers the place phase.",
            "coverage_notes": ["Add place-phase successes and corrected failures."],
            "risk_notes": ["Failure evidence is concentrated in grasp."],
            "recommended_collection": {
                "mode": "A_generalization",
                "focus_phases": ["place"],
                "target_count": 12,
            },
            "confidence": 0.81,
        }


def test_dataset_health_agent_writes_deterministic_report_and_llm_understanding(tmp_path: Path) -> None:
    selected_rollouts = [
        {
            "rollout_id": "r_success",
            "decision": "accepted",
            "final_success": True,
            "task_description": "pick cup",
            "phase_timeline": [{"phase": "approach"}, {"phase": "grasp"}],
        },
        {
            "rollout_id": "r_failure",
            "decision": "accepted",
            "final_success": False,
            "failure_analysis": {
                "failure_events": [
                    {"phase": "grasp", "failure_type": "slip"},
                ]
            },
        },
    ]
    result = DatasetHealthAgent(llm_client=_FakeDatasetHealthLLM()).analyze(
        selected_rollouts=selected_rollouts,
        task_config={"phases": ["approach", "grasp", "place"]},
        output_dir=tmp_path,
    )

    assert result.report["schema_version"] == "RoboLineage.dataset_health.v1"
    assert result.report["selected_rollout_count"] == 2
    assert result.report["success_count"] == 1
    assert result.report["failure_type_counts"] == {"slip": 1}
    assert result.report["phase_coverage"]["missing_phases"] == ["place"]
    assert result.report["recommended_action"] == "collect_phase_coverage"
    assert result.understanding["schema_version"] == "RoboLineage.dataset_health_understanding.v1"
    assert result.understanding["status"] == "generated"
    assert result.understanding["summary"] == "Dataset is useful but under-covers the place phase."

    assert result.report_path == tmp_path / "dataset_health_report.json"
    assert result.understanding_path == tmp_path / "dataset_health_understanding.json"
    assert result.report_markdown_path == tmp_path / "dataset_health_report.md"
    written = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert written["llm_understanding"]["status"] == "generated"
    assert (tmp_path / "dataset_health_events.jsonl").exists()


def test_dataset_health_agent_is_optional_when_llm_missing(tmp_path: Path) -> None:
    result = DatasetHealthAgent(enable_env_llm=False).analyze(
        selected_rollouts=[],
        task_config={"phases": ["approach"]},
        output_dir=tmp_path,
    )

    assert result.report["status"] == "insufficient_data"
    assert result.report["recommended_action"] == "collect_more_data"
    assert result.understanding["status"] == "not_configured"
    assert result.report["llm_understanding"]["status"] == "not_configured"
