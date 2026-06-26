from __future__ import annotations

import json
from pathlib import Path

from robolineage_shared_agents.master import MasterAgent
from robolineage_shared_agents.master.agent import OpenAICompatibleMasterLLMClient


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_master_agent_writes_state_memory_events_and_review(tmp_path: Path) -> None:
    task = tmp_path / "tasks" / "task_pick"
    _write_json(
        task / "task_manifest.json",
        {"task_description": "pick cup", "robot": {"robot_id": "arx_one"}},
    )
    (task / "task_config.latest.yaml").write_text(
        "task_description: pick cup\nphases:\n  - approach\n  - grasp\n  - place\n",
        encoding="utf-8",
    )
    _write_json(task / "rollouts" / "r1" / "dataset_admission.json", {"decision": "accepted"})
    _write_json(task / "training_runs" / "run1" / "training_status.json", {"status": "completed"})
    _write_json(
        task / "deployment_sessions" / "eval1" / "deployment_decision.json",
        {"decision": "collect_more_data", "gating_result": "fail"},
    )

    result = MasterAgent(enable_env_llm=False).review(task_root=task, health_summary={"status": "ok"})

    assert result.state_path.exists()
    assert result.memory_path.exists()
    assert result.events_path.exists()
    assert result.review_path.exists()
    assert result.report_path.exists()
    assert result.understanding_path.exists()
    assert result.understanding_report_path.exists()

    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    assert state["current_stage"] == "deployment_governance"
    assert state["next_action"]["action"] == "collect_more_data"
    assert state["health"]["status"] == "ok"
    assert state["robot"]["robot_id"] == "arx_one"
    assert state["llm_understanding"]["status"] == "not_configured"

    understanding = json.loads(result.understanding_path.read_text(encoding="utf-8"))
    assert understanding["status"] == "not_configured"

    events = [json.loads(line) for line in result.events_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == [
        "master_started",
        "artifacts_scanned",
        "llm_understanding_not_configured",
        "understanding_written",
        "state_written",
        "memory_updated",
        "review_written",
    ]


class _FakeMasterLLMClient:
    model = "fake-master-llm"

    def generate(self, context: dict) -> dict:
        assert context["state"]["current_stage"] == "deployment_governance"
        return {
            "summary": "The latest deployment gate failed and more data is needed.",
            "operator_brief": "Collect more focused rollouts before another training attempt.",
            "risk_interpretation": [
                {"severity": "medium", "code": "deployment_gate_failed", "reason": "The gate failed."},
            ],
            "suggested_next_action": {
                "action": "collect_more_data",
                "reason": "Evaluation failed the deployment gate.",
                "confidence": 0.82,
            },
            "memory_updates": ["Deployment gate failed; prioritize focused collection."],
        }


def test_master_agent_writes_llm_understanding_when_client_is_configured(tmp_path: Path) -> None:
    task = tmp_path / "tasks" / "task_pick"
    _write_json(
        task / "task_manifest.json",
        {"task_description": "pick cup", "robot": {"robot_id": "arx_one"}},
    )
    _write_json(
        task / "deployment_sessions" / "eval1" / "deployment_decision.json",
        {"decision": "collect_more_data", "gating_result": "fail"},
    )

    result = MasterAgent(llm_client=_FakeMasterLLMClient()).review(
        task_root=task,
        health_summary={"status": "ok"},
    )

    understanding = json.loads(result.understanding_path.read_text(encoding="utf-8"))
    assert understanding["schema_version"] == "RoboLineage.master_understanding.v1"
    assert understanding["status"] == "generated"
    assert understanding["model"] == "fake-master-llm"
    assert understanding["summary"] == "The latest deployment gate failed and more data is needed."
    assert understanding["suggested_next_action"]["action"] == "collect_more_data"

    review = json.loads(result.review_path.read_text(encoding="utf-8"))
    assert review["summary"] == "Collect more focused rollouts before another training attempt."
    assert review["llm_understanding"]["status"] == "generated"
    assert review["llm_understanding"]["path"] == str(result.understanding_path)

    events = [json.loads(line)["event"] for line in result.events_path.read_text(encoding="utf-8").splitlines()]
    assert "llm_understanding_completed" in events
    assert "understanding_written" in events


def test_master_llm_client_uses_master_route_env_with_sonnet_default(monkeypatch) -> None:
    monkeypatch.setattr("robolineage_shared_agents.llm_routes._load_dotenv_if_available", lambda: None)
    for name in (
        "MASTER_LLM_API_KEY",
        "MASTER_LLM_MODEL",
        "MASTER_LLM_BASE_URL",
        "ROBOLINEAGE_AGENT_API_KEY",
        "ROBOLINEAGE_AGENT_MODEL",
        "ROBOLINEAGE_AGENT_BASE_URL",
        "TASK_LLM_API_KEY",
        "TASK_LLM_MODEL",
        "TASK_LLM_BASE_URL",
        "VSA_VLM_API_KEY",
        "VSA_VLM_MODEL",
        "VSA_VLM_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MASTER_LLM_API_KEY", "master-key")
    monkeypatch.setenv("MASTER_LLM_BASE_URL", "https://gateway.example/v1")

    client = OpenAICompatibleMasterLLMClient.from_env()

    assert client is not None
    assert client.api_key == "master-key"
    assert client.model == "anthropic/claude-sonnet-4.6"
    assert client.base_url == "https://gateway.example/v1"


def test_master_llm_client_can_reuse_shared_agent_env(monkeypatch) -> None:
    monkeypatch.setattr("robolineage_shared_agents.llm_routes._load_dotenv_if_available", lambda: None)
    for name in (
        "MASTER_LLM_API_KEY",
        "MASTER_LLM_MODEL",
        "MASTER_LLM_BASE_URL",
        "ROBOLINEAGE_AGENT_API_KEY",
        "ROBOLINEAGE_AGENT_MODEL",
        "ROBOLINEAGE_AGENT_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROBOLINEAGE_AGENT_API_KEY", "agent-key")
    monkeypatch.setenv("ROBOLINEAGE_AGENT_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setenv("ROBOLINEAGE_AGENT_BASE_URL", "https://agent.example/v1")

    client = OpenAICompatibleMasterLLMClient.from_env()

    assert client is not None
    assert client.api_key == "agent-key"
    assert client.model == "anthropic/claude-sonnet-4.6"
    assert client.base_url == "https://agent.example/v1"
