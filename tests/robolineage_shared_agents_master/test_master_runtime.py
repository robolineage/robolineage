from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from robolineage_data_source.config.schema import (
    Config,
    HealthConfig,
    RecorderConfig,
    RolloutConfig,
    ServicesToggle,
    TuningConfig,
    VlmConfig,
    VsaConfig,
)


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _runtime_config(task_root: Path) -> Config:
    return Config(
        rollout=RolloutConfig(task_id=task_root.name, operator_id="op"),
        recorder=RecorderConfig(output_dir=str(task_root / "rollouts")),
        services=ServicesToggle(
            data_source=False,
            session=True,
            vsa=False,
            post_review=False,
            health_check=False,
        ),
        tuning=TuningConfig(),
        vsa=VsaConfig(),
        vlm=VlmConfig(),
        health=HealthConfig(port=8081),
    )


def test_unified_runtime_exposes_master_review_and_status(tmp_path: Path) -> None:
    from robolineage_app.runtime import UnifiedRuntime

    task_root = tmp_path / "task_pick"
    task_root.mkdir()
    (task_root / "task_manifest.json").write_text(
        '{"task_description":"pick cup","robot":{"robot_id":"arx_one"}}',
        encoding="utf-8",
    )
    runtime = UnifiedRuntime(_runtime_config(task_root))
    runtime.start()
    try:
        client = TestClient(runtime.session_app)

        review = client.post("/master/review")
        assert review.status_code == 200
        payload = review.json()
        assert payload["available"] is True
        assert payload["state"]["current_stage"] == "robot_onboarding"
        assert payload["state"]["robot"]["robot_id"] == "arx_one"
        assert payload["review"]["next_action"]["action"] == "define_task"

        status = client.get("/master/status")
        assert status.status_code == 200
        status_payload = status.json()
        assert status_payload["state"]["current_stage"] == "robot_onboarding"
        assert "MASTER_LLM" in status_payload["ai_routes"]["routes"]
        assert "api_key" not in status_payload["ai_routes"]["routes"]["MASTER_LLM"]
        assert (task_root / "master" / "master_state.json").exists()
    finally:
        runtime.stop_all()


def test_unified_runtime_auto_reviews_after_robot_onboarding(tmp_path: Path, monkeypatch) -> None:
    from robolineage_app.runtime import UnifiedRuntime

    profiles_root = tmp_path / "robot_profiles"
    monkeypatch.setenv("ROBOLINEAGE_ROBOT_PROFILES_ROOT", str(profiles_root))
    task_root = tmp_path / "task_pick"
    source_yaml = Path("configs/robot_profiles/arx_one_default.yaml").read_text(encoding="utf-8")
    runtime = UnifiedRuntime(_runtime_config(task_root))
    runtime.start()
    try:
        client = TestClient(runtime.session_app)
        response = client.post(
            "/robots/onboarding",
            json={"profile_yaml": source_yaml, "robot_note": "operator pasted default profile"},
        )
        assert response.status_code == 200
        assert response.json()["master_review"]["status"] == "queued"

        assert _wait_until(
            lambda: client.get("/master/status").json().get("available") is True,
            timeout=3.0,
        )
        status = client.get("/master/status")
        assert status.status_code == 200
        payload = status.json()
        assert payload["available"] is True
        assert payload["last_review_trigger"] == "robot_onboarding_completed"
        assert payload["master_queue"]["pending_count"] == 0
        assert payload["state"]["current_stage"] == "robot_onboarding"
        assert payload["state"]["robot"]["robot_id"] == "arx_one_default"
        assert (task_root / "master" / "master_state.json").exists()
    finally:
        runtime.stop_all()


def test_master_auto_review_is_enqueued_without_blocking(tmp_path: Path, monkeypatch) -> None:
    from robolineage_shared_agents.master.agent import MasterAgent
    from robolineage_app.runtime import UnifiedRuntime

    profiles_root = tmp_path / "robot_profiles"
    monkeypatch.setenv("ROBOLINEAGE_ROBOT_PROFILES_ROOT", str(profiles_root))
    task_root = tmp_path / "task_pick"
    source_yaml = Path("configs/robot_profiles/arx_one_default.yaml").read_text(encoding="utf-8")
    original_review = MasterAgent.review
    release_review = threading.Event()

    def slow_review(self, *, task_root, health_summary, trigger):
        release_review.wait(timeout=2.0)
        return original_review(self, task_root=task_root, health_summary=health_summary, trigger=trigger)

    monkeypatch.setattr(MasterAgent, "review", slow_review)
    runtime = UnifiedRuntime(_runtime_config(task_root))
    runtime.start()
    try:
        client = TestClient(runtime.session_app)
        completed = threading.Event()
        response_holder: dict[str, object] = {}

        def post_onboarding() -> None:
            response_holder["response"] = client.post(
                "/robots/onboarding",
                json={"profile_yaml": source_yaml, "robot_note": "operator pasted default profile"},
            )
            completed.set()

        thread = threading.Thread(target=post_onboarding)
        thread.start()
        finished_before_review = completed.wait(timeout=0.25)
        release_review.set()
        thread.join(timeout=2.0)

        assert finished_before_review is True
        response = response_holder["response"]
        assert response.status_code == 200
        payload = response.json()
        assert payload["master_review"]["status"] == "queued"
        assert payload["master_review"]["trigger"] == "robot_onboarding_completed"
    finally:
        release_review.set()
        runtime.stop_all()


def test_master_auto_review_debounces_duplicate_triggers(tmp_path: Path, monkeypatch) -> None:
    from robolineage_shared_agents.master.agent import MasterAgent
    from robolineage_app.runtime import UnifiedRuntime

    task_root = tmp_path / "task_pick"
    task_root.mkdir()
    (task_root / "task_manifest.json").write_text(
        '{"task_description":"pick cup","robot":{"robot_id":"arx_one"}}',
        encoding="utf-8",
    )
    original_review = MasterAgent.review
    review_started = threading.Event()
    release_review = threading.Event()
    calls: list[str] = []

    def slow_review(self, *, task_root, health_summary, trigger):
        calls.append(trigger)
        review_started.set()
        release_review.wait(timeout=2.0)
        return original_review(self, task_root=task_root, health_summary=health_summary, trigger=trigger)

    monkeypatch.setattr(MasterAgent, "review", slow_review)
    runtime = UnifiedRuntime(_runtime_config(task_root))
    runtime.start()
    try:
        first = runtime._enqueue_master_review("task_config_updated")
        assert review_started.wait(timeout=1.0)
        second = runtime._enqueue_master_review("task_config_updated")

        assert first["status"] == "queued"
        assert second["status"] == "debounced"
        queue = runtime.master_status()["master_queue"]
        assert queue["running"] is True
        assert queue["current_trigger"] == "task_config_updated"
        assert queue["pending_count"] == 0
    finally:
        release_review.set()
        runtime.stop_all()

    assert calls == ["task_config_updated"]
