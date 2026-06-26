from __future__ import annotations

import json
from pathlib import Path

from robolineage_train.monitor import TrainingMonitorAgent


class _FakeTrainingMonitorLLM:
    model = "fake-training-monitor-llm"

    def generate(self, context: dict) -> dict:
        assert context["deterministic_report"]["status"] == "unstable"
        assert "loss=nan" in context["log_excerpt"]
        return {
            "diagnosis": "Training became numerically unstable after the second step.",
            "likely_causes": ["NaN loss indicates optimizer or data instability"],
            "recommended_action": "inspect_training_instability",
            "operator_brief": "Stop this run and inspect the unstable batch or learning rate.",
            "confidence": 0.84,
        }


def test_training_monitor_agent_adds_optional_llm_understanding(tmp_path: Path) -> None:
    result = TrainingMonitorAgent(llm_client=_FakeTrainingMonitorLLM()).analyze(
        "step=1 loss=0.2\nstep=2 loss=nan\n",
        patterns={},
        output_dir=tmp_path,
    )

    assert result.report["status"] == "unstable"
    assert result.report["recommended_action"] == "inspect_training_instability"
    assert result.understanding["schema_version"] == "RoboLineage.training_monitor_understanding.v1"
    assert result.understanding["status"] == "generated"
    assert result.understanding["model"] == "fake-training-monitor-llm"
    assert result.understanding["diagnosis"] == "Training became numerically unstable after the second step."
    assert result.understanding_path == tmp_path / "training_monitor_understanding.json"
    assert result.report_path == tmp_path / "training_monitor_report.json"

    written = json.loads(result.understanding_path.read_text(encoding="utf-8"))
    assert written["recommended_action"] == "inspect_training_instability"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["llm_understanding"]["status"] == "generated"


def test_training_monitor_agent_keeps_deterministic_report_when_llm_missing(tmp_path: Path) -> None:
    result = TrainingMonitorAgent(enable_env_llm=False).analyze(
        "step=3 loss=0.1\nsaved checkpoint to /tmp/policy.ckpt\n",
        patterns={},
        output_dir=tmp_path,
    )

    assert result.report["status"] == "completed"
    assert result.report["checkpoints"] == ("/tmp/policy.ckpt",)
    assert result.understanding["status"] == "not_configured"
    assert result.report["llm_understanding"]["status"] == "not_configured"
