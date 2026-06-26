import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from robolineage_ar.video_source import SyntheticVideoSource
from robolineage_contracts.session import FeedbackEventName, SessionState
from robolineage_session.api import create_app
from robolineage_session.session import SessionRegistry


def _client(tmp_path: Path) -> tuple[TestClient, SessionRegistry]:
    registry = SessionRegistry()
    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=registry,
    )
    return TestClient(app), registry


def _start(client: TestClient, mode: str = "B1"):
    return client.post("/events", json={
        "event": "START_COLLECTING",
        "payload": {
            "task_id": "task_1",
            "mode": mode,
            "operator_id": "op",
            "policy_version": "1.0.0" if mode in {"B1", "B2"} else None,
        },
    })


def test_start_collecting_creates_session_and_metadata(tmp_path: Path):
    client, registry = _client(tmp_path)

    response = _start(client, mode="B1")

    assert response.status_code == 200
    assert response.json()["state"] == SessionState.COLLECTING.value
    session = registry.require_current()
    assert (session.rollout_dir / "metadata.json").exists()
    assert (session.rollout_dir / "events.jsonl").exists()


def test_state_endpoint_idle_without_session(tmp_path: Path):
    client, _ = _client(tmp_path)

    response = client.get("/state")

    assert response.json() == {"state": "IDLE", "rollout_id": None, "mode": None}


def test_pause_resume_stop_submit_flow_archives_outputs(tmp_path: Path):
    client, registry = _client(tmp_path)
    _start(client, mode="B1")
    session = registry.require_current()
    session.runtime_dir.mkdir(parents=True, exist_ok=True)
    (session.runtime_dir / "snapshots.jsonl").write_text('{"ok": true}\n', encoding="utf-8")

    assert client.post("/events", json={"event": "PAUSE_COLLECTING"}).json()["state"] == "PAUSED"
    assert client.post("/events", json={"event": "RESUME_COLLECTING"}).json()["state"] == "COLLECTING"
    assert client.post("/events", json={"event": "STOP_COLLECTING"}).json()["state"] == "REVIEWING"
    submit = client.post("/events", json={"event": "SUBMIT_ROLLOUT"})

    assert submit.status_code == 200
    assert submit.json()["state"] == "IDLE"
    assert (session.rollout_dir / ".closed").exists()
    assert (session.rollout_dir / "snapshots.jsonl").read_text(encoding="utf-8") == '{"ok": true}\n'
    assert registry.current() is None


def test_submit_events_share_rollout_id(tmp_path: Path):
    client, registry = _client(tmp_path)
    _start(client, mode="A")
    session = registry.require_current()
    client.post("/events", json={"event": "STOP_COLLECTING"})
    client.post("/events", json={"event": "SUBMIT_ROLLOUT"})

    rows = [
        json.loads(line)
        for line in (session.rollout_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert {row["rollout_id"] for row in rows} == {session.rollout_id}
    assert rows[-1]["event"] == FeedbackEventName.SESSION_CLOSED.value


def test_submit_archive_failure_keeps_session_reviewing(tmp_path: Path):
    client, registry = _client(tmp_path)
    _start(client, mode="A")
    session = registry.require_current()
    client.post("/events", json={"event": "STOP_COLLECTING"})
    session.rollout_dir.mkdir(parents=True, exist_ok=True)
    (session.rollout_dir / "snapshots.jsonl").write_text("old\n", encoding="utf-8")

    response = client.post("/events", json={"event": "SUBMIT_ROLLOUT"})

    assert response.status_code == 500
    assert response.json()["code"] == "E_SESSION_CLOSE_FAILED"
    assert session.state == SessionState.REVIEWING
    assert registry.current() == session


def test_trajectory_gate_accepts_only_collecting_policy_modes(tmp_path: Path):
    client, _ = _client(tmp_path)

    _start(client, mode="A")
    assert client.post("/trajectory", json={"points": [{"x": 0, "y": 0, "z": 1}]}).status_code == 204

    client.post("/events", json={"event": "STOP_COLLECTING"})
    client.post("/events", json={"event": "SUBMIT_ROLLOUT"})
    _start(client, mode="B1")
    accepted = client.post("/trajectory", json={"points": [{"x": 0, "y": 0, "z": 1}]})
    assert accepted.status_code == 200
    assert accepted.json()["accepted"] == 1

    client.post("/events", json={"event": "PAUSE_COLLECTING"})
    assert client.post("/trajectory", json={"points": [{"x": 0, "y": 0, "z": 1}]}).status_code == 204


def test_task_rollout_start_stop_callbacks(tmp_path: Path):
    calls: list[str] = []
    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=SessionRegistry(),
        on_rollout_start=lambda: calls.append("start") or {
            "active": True,
            "rollout_id": "r1",
            "rollout_dir": str(tmp_path / "rollouts" / "r1"),
        },
        on_rollout_stop=lambda: calls.append("stop") or {
            "active": False,
            "rollout_id": "r1",
        },
        on_rollout_state=lambda: {"active": False, "rollout_id": "r1"},
    )
    client = TestClient(app)

    started = client.post("/task/rollout/start")
    stopped = client.post("/task/rollout/stop")
    state = client.get("/task/rollout/state")

    assert started.status_code == 200
    assert started.json()["active"] is True
    assert stopped.status_code == 200
    assert stopped.json()["active"] is False
    assert state.json() == {"active": False, "rollout_id": "r1"}
    assert calls == ["start", "stop"]


def test_post_review_callbacks_are_exposed(tmp_path: Path):
    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=SessionRegistry(),
        on_post_review_state=lambda: {"active": True, "queue_size": 1},
        on_post_review_rollouts=lambda limit: {
            "status": {"active": True, "queue_size": 1},
            "rollouts": [{"rollout_id": "r1", "status": "completed"}],
            "limit": limit,
        },
        on_post_review_detail=lambda rollout_id: {
            "rollout": {"rollout_id": rollout_id, "status": "completed"},
            "failure_analysis": {"candidate_count": 0},
        },
    )
    client = TestClient(app)

    status = client.get("/post-review/status")
    rollouts = client.get("/post-review/rollouts?limit=7")
    detail = client.get("/post-review/rollouts/r1")

    assert status.status_code == 200
    assert status.json()["queue_size"] == 1
    assert rollouts.status_code == 200
    assert rollouts.json()["limit"] == 7
    assert rollouts.json()["rollouts"][0]["rollout_id"] == "r1"
    assert detail.status_code == 200
    assert detail.json()["rollout"]["status"] == "completed"


def test_robot_profile_callbacks_are_exposed(tmp_path: Path):
    calls: list[str] = []
    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=SessionRegistry(),
        on_robots=lambda: {
            "active_robot_id": "robot_a",
            "profiles": [{"robot_id": "robot_a", "display_name": "Robot A", "active": True}],
        },
        on_robot_detail=lambda robot_id: {
            "profile": {"robot_id": robot_id, "display_name": "Robot A"},
            "payload": {"schema_version": "RoboLineage.robot_profile.v1"},
            "validation": {"status": "ok", "streams": []},
        },
        on_robot_activate=lambda robot_id: calls.append(f"activate:{robot_id}") or {
            "status": "activated",
            "profile": {"robot_id": robot_id},
        },
        on_robot_validate=lambda robot_id: calls.append(f"validate:{robot_id}") or {
            "status": "ok",
            "robot_id": robot_id,
            "streams": [],
        },
    )
    client = TestClient(app)

    profiles = client.get("/robots")
    detail = client.get("/robots/robot_a")
    activated = client.post("/robots/robot_a/activate")
    validated = client.post("/robots/robot_a/validate")

    assert profiles.status_code == 200
    assert profiles.json()["active_robot_id"] == "robot_a"
    assert detail.status_code == 200
    assert detail.json()["profile"]["robot_id"] == "robot_a"
    assert activated.json()["status"] == "activated"
    assert validated.json()["status"] == "ok"
    assert calls == ["activate:robot_a", "validate:robot_a"]


def test_rollout_session_callbacks_are_exposed(tmp_path: Path):
    calls: list[str] = []
    active = {
        "active": True,
        "kind": "collection",
        "session_id": "collection_s1",
        "rollout_count": 0,
        "rollout_ids": [],
    }

    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=SessionRegistry(),
        on_rollout_session_state=lambda: active,
        on_collection_session_start=lambda: calls.append("collection_start") or active,
        on_collection_session_stop=lambda: calls.append("collection_stop") or {"active": False, "kind": None},
        on_deployment_session_start=lambda body: calls.append(f"deployment_start:{body.get('policy_version')}") or {
            "active": True,
            "kind": "deployment",
            "session_id": "deploy_s1",
            "policy_version": body.get("policy_version"),
            "rollout_count": 0,
            "rollout_ids": [],
        },
        on_deployment_session_stop=lambda: calls.append("deployment_stop") or {"active": False, "kind": None},
    )
    client = TestClient(app)

    state = client.get("/task/session/state")
    collection_started = client.post("/task/session/collection/start")
    collection_stopped = client.post("/task/session/collection/stop")
    deployment_started = client.post(
        "/task/session/deployment/start",
        json={"policy_version": "policy_1"},
    )
    deployment_stopped = client.post("/task/session/deployment/stop")

    assert state.status_code == 200
    assert state.json()["session_id"] == "collection_s1"
    assert collection_started.json()["kind"] == "collection"
    assert collection_stopped.json()["active"] is False
    assert deployment_started.json()["policy_version"] == "policy_1"
    assert deployment_stopped.json()["active"] is False
    assert calls == [
        "collection_start",
        "collection_stop",
        "deployment_start:policy_1",
        "deployment_stop",
    ]


def test_training_framework_callbacks_are_exposed(tmp_path: Path):
    calls: list[str] = []
    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=SessionRegistry(),
        on_training_framework_state=lambda: {"active": False, "current_run": None},
        on_training_framework_runs=lambda limit: {
            "status": {"active": False},
            "runs": [{"run_id": "train_1", "status": "completed"}],
            "limit": limit,
        },
        on_training_framework_detail=lambda run_id: {
            "run": {"run_id": run_id, "status": "completed"},
            "deployment_recommendation": {"decision": "deploy_recommended"},
        },
        on_training_framework_run_demo=lambda: calls.append("demo") or {
            "status": "started",
            "run_id": "train_1",
        },
        on_training_framework_discover=lambda body: {
            "status": "generated",
            "repo_root": body["repo_root"],
        },
    )
    client = TestClient(app)

    status = client.get("/training-framework/status")
    runs = client.get("/training-framework/runs?limit=5")
    detail = client.get("/training-framework/runs/train_1")
    started = client.post("/training-framework/run-demo")
    discovered = client.post("/training-framework/discover", json={"repo_root": str(tmp_path)})

    assert status.status_code == 200
    assert status.json()["active"] is False
    assert runs.status_code == 200
    assert runs.json()["limit"] == 5
    assert detail.status_code == 200
    assert detail.json()["deployment_recommendation"]["decision"] == "deploy_recommended"
    assert started.status_code == 200
    assert started.json()["status"] == "started"
    assert discovered.status_code == 200
    assert discovered.json()["status"] == "generated"
    assert calls == ["demo"]


def test_user_input_value_errors_return_400(tmp_path: Path):
    def _raise_value_error(*_args, **_kwargs):
        raise ValueError("bad user input")

    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=SessionRegistry(),
        on_training_framework_discover=_raise_value_error,
        on_training_selection_create=_raise_value_error,
        on_training_run_start=_raise_value_error,
    )
    client = TestClient(app)

    discovered = client.post("/training-framework/discover", json={})
    selection = client.post("/tasks/task_a/training-selections", json={})
    run = client.post("/tasks/task_a/training-runs", json={})

    assert discovered.status_code == 400
    assert selection.status_code == 400
    assert run.status_code == 400
    assert discovered.json()["error"] == "bad user input"
    assert selection.json()["error"] == "bad user input"
    assert run.json()["error"] == "bad user input"


def test_task_lifecycle_callbacks_are_exposed(tmp_path: Path):
    calls: list[str] = []
    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=SessionRegistry(),
        on_tasks=lambda: {
            "active_task_id": "task_a",
            "tasks": [{"task_id": "task_a", "display_name": "stack blocks"}],
        },
        on_task_create=lambda body: calls.append(f"create:{body.get('task_description')}") or {
            "status": "activated",
            "task": {"task_id": "task_b", "display_name": body.get("task_description")},
        },
        on_task_activate=lambda task_id: calls.append(f"activate:{task_id}") or {
            "status": "activated",
            "task": {"task_id": task_id},
        },
        on_task_detail=lambda task_id: {"task": {"task_id": task_id}, "policies": []},
        on_task_collection_sessions=lambda task_id: {
            "task_id": task_id,
            "sessions": [{"session_id": "collection_1", "rollout_count": 2}],
        },
        on_task_collection_session_detail=lambda task_id, session_id: {
            "session": {"session_id": session_id},
            "summary": {"task_id": task_id, "rollout_count": 2},
        },
        on_task_deployment_sessions=lambda task_id: {
            "task_id": task_id,
            "sessions": [{"session_id": "deployment_1", "policy_version": "policy_1", "rollout_count": 3}],
        },
        on_task_deployment_session_detail=lambda task_id, session_id: {
            "session": {"session_id": session_id, "policy_version": "policy_1"},
            "summary": {"task_id": task_id, "rollout_count": 3},
            "deployment_decision": {"decision": "collect_more_data"},
        },
        on_training_selections=lambda task_id: {
            "task_id": task_id,
            "selections": [{"selection_id": "selection_1", "selected_rollout_count": 2}],
        },
        on_training_selection_create=lambda task_id, body: calls.append(f"selection:{task_id}") or {
            "status": "created",
            "selection": {
                "selection_id": "selection_2",
                "rollout_ids": body.get("rollout_ids") or [],
            },
        },
        on_framework_profiles=lambda task_id: {
            "task_id": task_id,
            "profiles": [{"profile_id": "profile_1"}],
        },
        on_training_run_start=lambda task_id, body: calls.append(f"train:{task_id}:{body.get('policy_version')}") or {
            "status": "started",
            "request_id": "train_request_1",
        },
        on_policies=lambda task_id: {
            "task_id": task_id,
            "policies": [{"policy_version": "policy_1"}],
        },
    )
    client = TestClient(app)

    assert client.get("/tasks").json()["active_task_id"] == "task_a"
    assert client.post("/tasks", json={"task_description": "place brick"}).json()["task"]["task_id"] == "task_b"
    assert client.post("/tasks/task_a/activate").json()["task"]["task_id"] == "task_a"
    assert client.get("/tasks/task_a").json()["task"]["task_id"] == "task_a"
    assert client.get("/tasks/task_a/collection-sessions").json()["sessions"][0]["session_id"] == "collection_1"
    assert client.get("/tasks/task_a/collection-sessions/collection_1").json()["summary"]["rollout_count"] == 2
    assert client.get("/tasks/task_a/deployment-sessions").json()["sessions"][0]["session_id"] == "deployment_1"
    assert client.get("/tasks/task_a/deployment-sessions/deployment_1").json()["deployment_decision"]["decision"] == "collect_more_data"
    assert client.get("/tasks/task_a/training-selections").json()["selections"][0]["selection_id"] == "selection_1"
    created = client.post(
        "/tasks/task_a/training-selections",
        json={"rollout_ids": ["r1"], "include_decisions": ["accepted"]},
    )
    assert created.json()["selection"]["selection_id"] == "selection_2"
    assert client.get("/tasks/task_a/framework-profiles").json()["profiles"][0]["profile_id"] == "profile_1"
    started = client.post(
        "/tasks/task_a/training-runs",
        json={"selection_id": "selection_2", "framework_profile_id": "profile_1", "policy_version": "policy_1"},
    )
    assert started.json()["status"] == "started"
    assert client.get("/tasks/task_a/policies").json()["policies"][0]["policy_version"] == "policy_1"
    assert calls == [
        "create:place brick",
        "activate:task_a",
        "selection:task_a",
        "train:task_a:policy_1",
    ]


def test_legacy_task_stop_uses_rollout_stop_callback(tmp_path: Path):
    calls: list[str] = []
    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=SessionRegistry(),
        on_rollout_stop=lambda: calls.append("stop") or {"active": False},
    )
    client = TestClient(app)

    response = client.post("/task/stop")

    assert response.status_code == 200
    assert response.json()["active"] is False
    assert calls == ["stop"]


def test_ai_routes_status_endpoint_uses_callback() -> None:
    app = create_app(
        on_ai_routes_status=lambda: {
            "routes": {
                "ROBOT_ONBOARDING_LLM": {
                    "configured": False,
                    "implemented": True,
                }
            }
        }
    )
    client = TestClient(app)

    response = client.get("/ai/routes")

    assert response.status_code == 200
    assert response.json()["routes"]["ROBOT_ONBOARDING_LLM"]["implemented"] is True


def test_task_configure_sends_current_frame_to_llm(tmp_path: Path, monkeypatch):
    captured: dict = {}
    fake_payload = {
        "phases": ["approach_handle", "pull_drawer"],
        "hints": {"approach_handle": "Move to the visible handle."},
        "failure_signals": ["missed_handle"],
        "phase_action_hints": {},
        "phase_visual_hints": {"approach_handle": "The gripper is near the handle."},
    }

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(fake_payload))
                    )
                ]
            )

    class _FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("robolineage_shared_agents.llm_routes._load_dotenv_if_available", lambda: None)
    monkeypatch.setenv("ROBOLINEAGE_TASK_DIR", str(tmp_path))
    monkeypatch.setenv("TASK_LLM_BACKEND", "openai")
    monkeypatch.setenv("TASK_LLM_API_KEY", "task-key")
    monkeypatch.setenv("TASK_LLM_MODEL", "")
    monkeypatch.setenv("OPENAI_MODEL", "")

    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=SessionRegistry(),
        video_source=SyntheticVideoSource(color=(0, 0, 255)),
    )
    client = TestClient(app)

    response = client.post("/task/configure", json={"task_description": "open the drawer"})

    assert response.status_code == 200
    assert captured["model"] == "anthropic/claude-sonnet-4.6"
    user_content = captured["messages"][1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "text"
    assert "open the drawer" in user_content[0]["text"]
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert (tmp_path / "task_config.yaml").exists()
    payload = response.json()
    assert payload["task_config_version"] == "v1"
    assert (tmp_path / "task_configs" / "task_config.v1.yaml").exists()
    assert (tmp_path / "task_config.latest.yaml").exists()
    index = json.loads((tmp_path / "task_configs" / "task_config_index.json").read_text())
    assert index["latest_version"] == "v1"
    source = index["entries"][0]["source"]
    assert source["llm_backend"] == "openai"
    assert source["llm_model"] == "anthropic/claude-sonnet-4.6"
    assert source["llm_configured"] is True


def test_vsa_decisions_keeps_previous_rollout_until_active_log_has_rows(tmp_path: Path, monkeypatch):
    task_dir = tmp_path / "task"
    old_rollout = task_dir / "rollouts" / "old"
    old_log = old_rollout / "logs" / "tiaoshi.log"
    old_log.parent.mkdir(parents=True)
    old_log.write_text(
        json.dumps(
            {
                "timestamp": "old",
                "event_type": "periodic_sample",
                "anchor_frame": 1,
                "end_frame": 1,
                "n_images": 0,
                "image_paths": [],
                "prior": {},
                "parsed": {"phase": "old_phase", "progress": "advancing"},
                "raw_response": "{}",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    current_rollout = task_dir / "rollouts" / "current"
    monkeypatch.setenv("ROBOLINEAGE_TASK_DIR", str(task_dir))

    app = create_app(
        data_root=tmp_path / "data" / "rollouts",
        runtime_root=tmp_path / "runtime",
        registry=SessionRegistry(),
        on_rollout_state=lambda: {
            "active": True,
            "rollout_id": "current",
            "rollout_dir": str(current_rollout),
        },
    )
    client = TestClient(app)

    rows = client.get("/vsa/decisions").json()
    assert len(rows) == 1
    assert rows[0]["phase"] == "old_phase"

    current_log = current_rollout / "logs" / "tiaoshi.log"
    current_log.parent.mkdir(parents=True)
    current_log.write_text(
        json.dumps(
            {
                "timestamp": "current",
                "event_type": "periodic_sample",
                "anchor_frame": 2,
                "end_frame": 2,
                "n_images": 0,
                "image_paths": [],
                "prior": {},
                "parsed": {"phase": "current_phase", "progress": "advancing"},
                "raw_response": "{}",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = client.get("/vsa/decisions").json()
    assert len(rows) == 1
    assert rows[0]["phase"] == "current_phase"
