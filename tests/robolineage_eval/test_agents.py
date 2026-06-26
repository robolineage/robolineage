from __future__ import annotations

import json
from pathlib import Path

from robolineage_eval import DeploymentGovernanceAgent, PolicyEvaluationAgent
from robolineage_schemas import validate


class _FakeDeploymentGovernanceLLM:
    model = "fake-deployment-governance-llm"

    def generate(self, context: dict) -> dict:
        assert context["deterministic_decision"]["decision"] == "deploy_recommended"
        return {
            "summary": "Rule gate passes, but monitor the grasp boundary closely.",
            "risk_notes": ["One weak phase would require more evaluation before automatic deploy."],
            "llm_suggested_decision": "rollback_recommended",
            "operator_brief": "Keep the rule decision, but ask a human to inspect the weak phase.",
            "confidence": 0.77,
        }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rollout(path: Path, rows: list[dict]) -> Path:
    path.mkdir(parents=True)
    (path / "task_config.yaml").write_text(
        "\n".join(
            [
                "task_description: pick red block and place it on blue block",
                "phases:",
                "  - approach",
                "  - grasp",
                "  - place",
                "failure_signals:",
                "  - object dropped",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(path / "snapshots.jsonl", rows)
    return path


def _assert_schema_valid(path: Path, schema_name: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = [issue for issue in validate(payload, schema_name) if issue.severity == "error"]
    assert errors == []


def test_policy_evaluation_agent_writes_eval_not_dataset_admission(tmp_path: Path):
    rollout_dir = _rollout(
        tmp_path / "r_success",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 1.0,
                "frame_id": 10,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "grasp",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 2.0,
                "frame_id": 20,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
            },
        ],
    )

    result = PolicyEvaluationAgent(use_vlm=False).run(
        rollout_dir,
        policy_version="policy_1",
        evaluation_session_id="deploy_s1",
        evaluation_mode="deployment",
    )

    assert result.status == "completed"
    assert (rollout_dir / "policy_evaluation.json").exists()
    assert (rollout_dir / "collection_recommendation.json").exists()
    assert (rollout_dir / "eval_review_report.md").exists()
    assert not (rollout_dir / "dataset_admission.json").exists()
    evaluation = json.loads((rollout_dir / "policy_evaluation.json").read_text())
    assert evaluation["final_success"] is True
    assert evaluation["recommended_next_action"] == "continue_evaluation"
    _assert_schema_valid(rollout_dir / "policy_evaluation.json", "policy_evaluation")
    _assert_schema_valid(rollout_dir / "collection_recommendation.json", "collection_recommendation")


def test_deployment_governance_summarizes_multiple_rollouts(tmp_path: Path):
    success = _rollout(
        tmp_path / "r_success",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 1.0,
                "frame_id": 2,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "place",
                "imminent_failure": False,
                "confidence": 0.9,
                "needs_review": False,
                "raw_response": "{}",
            },
        ],
    )
    failure = _rollout(
        tmp_path / "r_failure",
        [
            {
                "timestamp": 0.0,
                "frame_id": 1,
                "progress": "advancing",
                "risk_level": "low",
                "phase": "approach",
                "imminent_failure": False,
                "confidence": 0.8,
                "needs_review": False,
                "raw_response": "{}",
            },
            {
                "timestamp": 1.0,
                "frame_id": 2,
                "progress": "stalled",
                "risk_level": "high",
                "phase": "grasp",
                "imminent_failure": True,
                "confidence": 0.7,
                "needs_review": True,
                "raw_response": "{}",
            },
        ],
    )
    agent = PolicyEvaluationAgent(use_vlm=False)
    agent.run(success, policy_version="policy_1", evaluation_session_id="deploy_s1")
    agent.run(failure, policy_version="policy_1", evaluation_session_id="deploy_s1")

    summary = DeploymentGovernanceAgent(enable_env_llm=False).summarize_session(
        rollout_dirs=[success, failure],
        output_dir=tmp_path / "deployment_sessions" / "deploy_s1",
        session_id="deploy_s1",
        policy_version="policy_1",
        mode="deployment",
        deploy_success_threshold=0.8,
        rollback_success_threshold=0.5,
    )

    assert summary["policy_eval_summary"]["rollout_count"] == 2
    assert summary["policy_eval_summary"]["success_rate"] == 0.5
    assert summary["deployment_decision"]["decision"] == "collect_more_data"
    assert summary["collection_recommendation"]["recommended_mode"] == "B2_deployment_monitoring"
    assert summary["next_collection_brief"]["target_count"] == 20
    assert "Collect 20 rollout" in summary["next_collection_brief"]["operator_brief"]
    session_dir = tmp_path / "deployment_sessions" / "deploy_s1"
    assert (session_dir / "deployment_decision.json").exists()
    assert (session_dir / "next_collection_brief.json").exists()
    _assert_schema_valid(session_dir / "deployment_decision.json", "deployment_decision")
    _assert_schema_valid(session_dir / "collection_recommendation.json", "collection_recommendation")
    _assert_schema_valid(session_dir / "next_collection_brief.json", "next_collection_brief")


def test_deployment_governance_llm_understanding_never_overrides_rule_decision(tmp_path: Path) -> None:
    rollout_a = tmp_path / "eval_rollouts" / "a"
    rollout_b = tmp_path / "eval_rollouts" / "b"
    for rollout_dir in (rollout_a, rollout_b):
        _write_json(
            rollout_dir / "policy_evaluation.json",
            {
                "rollout_id": rollout_dir.name,
                "final_success": True,
                "phases_seen": ["approach", "grasp", "place"],
                "failure_type_counts": {},
                "phase_weakness": [],
                "recommended_next_action": "continue_evaluation",
            },
        )

    summary = DeploymentGovernanceAgent(llm_client=_FakeDeploymentGovernanceLLM()).summarize_session(
        rollout_dirs=[rollout_a, rollout_b],
        output_dir=tmp_path / "deployment_sessions" / "deploy_llm",
        session_id="deploy_llm",
        policy_version="policy_2",
        deploy_success_threshold=0.8,
        rollback_success_threshold=0.5,
    )

    assert summary["deployment_decision"]["decision"] == "deploy_recommended"
    understanding = summary["deployment_governance_understanding"]
    assert understanding["schema_version"] == "RoboLineage.deployment_governance_understanding.v1"
    assert understanding["status"] == "generated"
    assert understanding["model"] == "fake-deployment-governance-llm"
    assert understanding["llm_suggested_decision"] == "rollback_recommended"
    assert understanding["deterministic_decision"] == "deploy_recommended"

    session_dir = tmp_path / "deployment_sessions" / "deploy_llm"
    assert (session_dir / "deployment_governance_understanding.json").exists()
    assert (session_dir / "deployment_governance_understanding.md").exists()
