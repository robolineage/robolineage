"""UnifiedRuntime end-to-end smoke tests (mac-safe, no ROS2)."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from robolineage_data_source.config.schema import (
    ArmTopicSpec,
    CameraConfig,
    CameraTopicSpec,
    Config,
    HealthConfig,
    PostReviewConfig,
    RecorderConfig,
    RolloutConfig,
    Ros2AdapterConfig,
    ServicesToggle,
    TuningConfig,
    VlmConfig,
    VsaConfig,
)


def _make_config(tmp_path) -> Config:
    """Build a Config that does not touch ROS2 in tests."""
    return Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        cameras={"cam_master": CameraConfig(type="mock_master")},
        recorder=RecorderConfig(output_dir=str(tmp_path)),
        services=ServicesToggle(
            data_source=True,
            session=True,
            vsa=False,
            health_check=True,
        ),
        tuning=TuningConfig(),
        vsa=VsaConfig(),
        vlm=VlmConfig(),        health=HealthConfig(port=8081),
    )


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _write_raw_artifacts(rollout_dir) -> None:
    raw = rollout_dir / "raw"
    bag_dir = raw / "rosbag2"
    bag_dir.mkdir(parents=True, exist_ok=True)
    (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n", encoding="utf-8")
    (raw / "raw_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "RoboLineage.raw_rosbag_manifest.v1",
                "status": "closed",
                "raw_format": "rosbag2",
                "bag_dir": str(bag_dir),
                "topics": ["/camera/camera_h/color/image_raw/compressed", "/arm_master_r_status"],
            }
        ),
        encoding="utf-8",
    )
    (raw / "metadata.json").write_text("{}", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _decode_first_mjpeg_frame(data: bytes):
    import cv2
    import numpy as np

    start = data.index(b"\xff\xd8")
    end = data.index(b"\xff\xd9", start) + 2
    return cv2.imdecode(np.frombuffer(data[start:end], dtype=np.uint8), cv2.IMREAD_COLOR)


def _ros_topic_adapter_config() -> Ros2AdapterConfig:
    return Ros2AdapterConfig(
        type="ros2_profile",
        ros_domain_id=0,
        cameras={
            "camera_h": CameraTopicSpec(
                topic="/cam/x/image/compressed",
                transport="compressed",
                qos="sensor_data",
                stream_id="camera_h",
                camera_name="camera_h",
            )
        },
        arms={
            "right_arm": ArmTopicSpec(
                slave_status="/robot/x/state",
                state_stream_id="right_arm",
                msg_type="example_msgs/msg/RobotState",
            )
        },
    )


class _FakeRawRecorder:
    def __init__(self, rollout_dir: Path) -> None:
        self.rollout_dir = rollout_dir

    def stop_capture(self) -> None:
        _write_raw_artifacts(self.rollout_dir)

    def finalize(self, *, outcome, note: str) -> None:
        _write_raw_artifacts(self.rollout_dir)
        manifest = self.rollout_dir / "raw" / "raw_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload.update({"status": "closed", "outcome": str(getattr(outcome, "value", outcome)), "note": note})
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _install_fake_ros_topic_vsa(monkeypatch) -> None:
    import robolineage_shared_agents.visual_snapshot.realtime as realtime_mod
    from robolineage_app.runtime import UnifiedRuntime

    def fake_run_ros_topic_stream(*, pipeline, stop_event=None, **kwargs):
        if stop_event is not None:
            stop_event.wait(timeout=1.0)
        pipeline.drain()
        pipeline.close()
        return []

    def fake_start_raw_recorder(self, rollout_dir):
        _write_raw_artifacts(Path(rollout_dir))
        return _FakeRawRecorder(Path(rollout_dir))

    monkeypatch.setattr(realtime_mod, "run_ros_topic_stream", fake_run_ros_topic_stream)
    monkeypatch.setattr(UnifiedRuntime, "_start_raw_recorder", fake_start_raw_recorder)


def test_training_framework_command_context_parser_extracts_two_box_input():
    from robolineage_app.runtime import _maybe_prefix_conda_run, _normalize_script_command, _parse_framework_command_context

    parsed = _parse_framework_command_context(
        "\n".join(
            [
                "repo_root: /tmp/policy_repo",
                "conda env: policy_env",
                "train command:",
                "  python train.py --dataset {dataset_output} --ckpt_dir {checkpoint_dir}",
                "checkpoint output: {checkpoint_dir}/*.ckpt",
                "train launch: external_terminal",
            ]
        )
    )

    assert parsed["repo_root"] == "/tmp/policy_repo"
    assert parsed["conda_env"] == "policy_env"
    assert parsed["train_command"] == "python train.py --dataset {dataset_output} --ckpt_dir {checkpoint_dir}"
    assert parsed["checkpoint_glob"] == "{checkpoint_dir}/*.ckpt"
    assert _maybe_prefix_conda_run(parsed["train_command"], parsed["conda_env"]).startswith("conda run -n policy_env ")
    assert _normalize_script_command("train.py --dataset {dataset_output}") == "python train.py --dataset {dataset_output}"


def test_training_framework_context_parser_repairs_wrapped_repo_root(tmp_path):
    from robolineage_app.runtime import _parse_framework_command_context

    repo = tmp_path / "Transengram_datacollection-main"
    (repo / "tools").mkdir(parents=True)
    (repo / "README.md").write_text("# training repo\n", encoding="utf-8")
    train_script = repo / "tools" / "02_train.sh"
    train_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    parsed = _parse_framework_command_context(
        "\n".join(
            [
                f"repo_root: {tmp_path}/",
                "  Transengram_datacollection-main",
                "    conda env: act",
                f"    train command: {train_script}",
            ]
        )
    )

    assert parsed["repo_root"] == str(repo)
    assert parsed["conda_env"] == "act"
    assert parsed["train_command"] == str(train_script)


def test_training_framework_remote_repo_root_validation():
    from robolineage_app.runtime import _normalize_remote_repo_command, _normalize_remote_repo_root

    assert _normalize_remote_repo_root("home/user/code/repo") == "/home/user/code/repo"
    with pytest.raises(ValueError, match="appears duplicated"):
        _normalize_remote_repo_root("home/user/code/repo/code/repo")
    with pytest.raises(ValueError, match="must be an absolute"):
        _normalize_remote_repo_root("relative/repo")
    local_repo = Path("/tmp/repo")
    assert (
        _normalize_remote_repo_command("/home/user/code/repo/tools/train.sh --flag x", "/home/user/code/repo", local_repo)
        == "tools/train.sh --flag x"
    )


def test_training_framework_discover_returns_trace_and_manifest(tmp_path, monkeypatch):
    from robolineage_app.runtime import UnifiedRuntime
    from robolineage_train.discovery import OpenAICompatibleDiscoveryClient

    monkeypatch.setattr(OpenAICompatibleDiscoveryClient, "from_env", classmethod(lambda cls: None))
    repo = tmp_path / "policy_repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Policy repo\n", encoding="utf-8")
    (repo / "train.py").write_text(
        "from pathlib import Path\n"
        "from torch.utils.tensorboard import SummaryWriter\n"
        "writer = SummaryWriter(log_dir='runs/train')\n"
        "Path('policy_best.ckpt').write_text('ckpt')\n",
        encoding="utf-8",
    )
    runtime = UnifiedRuntime(_make_config(tmp_path / "runtime"))

    result = runtime.training_framework_discover(
        {
            "repo_location": "local",
            "repo_root": str(repo),
            "target_dataset_format": "custom dataset consumed by train.py",
            "command_context": "train command: python train.py",
        }
    )

    assert result["status"] == "generated"
    assert result["events"]
    assert result["target_contract"]
    assert result["integration_manifest"]["schema_version"] == "RoboLineage.training_integration_manifest.v1"
    assert result["integration_manifest"]["outputs"]["primary_metrics_source"] in {"tensorboard", "stdout"}
    assert result["llm_understanding"]["status"] == "skipped"


def test_training_framework_discover_start_exposes_running_job_until_completion(tmp_path, monkeypatch):
    from robolineage_app.runtime import UnifiedRuntime
    from robolineage_train.discovery import OpenAICompatibleDiscoveryClient

    monkeypatch.setattr(OpenAICompatibleDiscoveryClient, "from_env", classmethod(lambda cls: None))
    repo = tmp_path / "policy_repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Policy repo\n", encoding="utf-8")
    (repo / "train.py").write_text("print('train')\n", encoding="utf-8")
    runtime = UnifiedRuntime(_make_config(tmp_path / "runtime"))

    job = runtime.training_framework_discover_start(
        {
            "repo_location": "local",
            "repo_root": str(repo),
            "target_dataset_format": "custom dataset consumed by train.py",
            "command_context": "train command: python train.py",
        }
    )

    assert job["status"] == "running"
    assert job["job_id"].startswith("discover_")
    status = job
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        status = runtime.training_framework_discovery_job(job["job_id"])
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)

    assert status["status"] == "completed", status
    assert status["result"]["integration_manifest"]["schema_version"] == "RoboLineage.training_integration_manifest.v1"
    assert any(item["event"] == "discovery_agent_completed" for item in status["events"])


def test_training_framework_detail_exposes_per_agent_understanding_artifacts(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    task_root = tmp_path / "task_lifecycle"
    run_dir = task_root / "training_runs" / "train_artifacts"
    framework_dir = run_dir / "framework"
    framework_dir.mkdir(parents=True)
    _write_json(run_dir / "training_run_config.json", {"policy_version": "policy_1"})
    _write_json(run_dir / "deployment_recommendation.json", {"decision": "collect_more_data"})
    _write_json(run_dir / "policy.ROBOLINEAGE_context.json", {"dataset": {"version_id": "v1"}})
    _write_json(run_dir / "dataset_health_report.json", {"status": "needs_more_coverage", "recommended_action": "collect_phase_coverage"})
    _write_json(run_dir / "dataset_health_understanding.json", {"status": "generated", "summary": "Under-covers place."})
    (run_dir / "dataset_health_report.md").write_text("# Dataset health\n", encoding="utf-8")
    _write_json(framework_dir / "training_result.json", {"policy_version": "policy_1"})
    _write_json(framework_dir / "training_status.json", {"status": "completed"})
    _write_json(framework_dir / "dataset_adapt_status.json", {"status": "completed"})
    _write_json(framework_dir / "dataset_adapt_result.json", {"status": "completed"})
    _write_json(framework_dir / "training_monitor_understanding.json", {"status": "not_configured", "diagnosis": "Training completed."})
    (framework_dir / "training_monitor_report.md").write_text("# Training monitor\n", encoding="utf-8")
    (run_dir / "train_manifest.jsonl").write_text('{"rollout_id":"r1"}\n', encoding="utf-8")

    runtime = UnifiedRuntime(_make_config(task_root / "rollouts"))
    detail = runtime.training_framework_detail("train_artifacts")

    assert detail["dataset_health_report"]["recommended_action"] == "collect_phase_coverage"
    assert detail["dataset_health_understanding"]["summary"] == "Under-covers place."
    assert detail["dataset_health_report_md"] == "# Dataset health\n"
    assert detail["training_monitor_understanding"]["diagnosis"] == "Training completed."
    assert detail["training_monitor_report_md"] == "# Training monitor\n"


def test_deployment_session_detail_exposes_governance_understanding(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    task_root = tmp_path / "task_lifecycle"
    session_dir = task_root / "deployment_sessions" / "deploy_artifacts"
    _write_json(
        session_dir / "policy_eval_summary.json",
        {"session_id": "deploy_artifacts", "policy_version": "policy_1", "rollout_ids": []},
    )
    _write_json(session_dir / "deployment_decision.json", {"decision": "collect_more_data"})
    _write_json(session_dir / "collection_recommendation.json", {"recommended_mode": "B2_deployment_monitoring"})
    _write_json(session_dir / "next_collection_brief.json", {"operator_brief": "Collect more deployment rollouts."})
    _write_json(
        session_dir / "deployment_governance_understanding.json",
        {
            "status": "generated",
            "summary": "Rule gate asks for more data.",
            "deterministic_decision": "collect_more_data",
        },
    )
    (session_dir / "deployment_governance_understanding.md").write_text("# Governance\n", encoding="utf-8")

    runtime = UnifiedRuntime(_make_config(task_root / "rollouts"))
    detail = runtime.task_deployment_session_detail(task_root.name, "deploy_artifacts")

    assert detail["deployment_governance_understanding"]["summary"] == "Rule gate asks for more data."
    assert detail["deployment_governance_understanding_report"] == "# Governance\n"


def test_unified_runtime_ar_overlay_uses_http_trajectory_api_not_bus(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    cfg = _make_config(tmp_path)
    cfg.services = ServicesToggle(
        data_source=False,
        session=True,
        vsa=False,
        post_review=False,
        health_check=False,
    )

    runtime = UnifiedRuntime(cfg)
    runtime.start()
    try:
        client = TestClient(runtime.session_app)
        assert not hasattr(runtime, "bus")

        opened = client.post(
            "/events",
            json={
                "event": "START_COLLECTING",
                "payload": {
                    "task_id": "t",
                    "mode": "B1",
                    "operator_id": "op",
                    "policy_version": "policy_1",
                },
            },
        )
        assert opened.status_code == 200

        accepted = client.post(
            "/trajectory",
            json={"points": [{"x": 0.1, "y": 0.2, "z": 0.3}, {"x": 0.2, "y": 0.2, "z": 0.4}]},
        )
        assert accepted.status_code == 200
        assert accepted.json()["accepted"] == 2
        assert client.get("/health").json()["trajectory_points"] == 2
    finally:
        runtime.stop_all()


def test_robot_onboard_validation_response_is_json_safe(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    cfg = Config(
        rollout=RolloutConfig(task_id="arx_one_default", operator_id="op"),
        robot_profile_path="configs/robot_profiles/arx_one_default.yaml",
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
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
    runtime = UnifiedRuntime(cfg)
    runtime.start()
    try:
        client = TestClient(runtime.session_app)

        response = client.post("/robots/arx_one_default/validate")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert [stream["present"] for stream in payload["streams"]] == [True, True]
        assert payload["streams"][0]["ros_topic"] == "/camera/camera_h/color/image_raw/compressed"
        assert payload["streams"][0]["sample_meta"] is None
        signals = payload["canonical_signals"]
        assert signals["primary_image"]["topic"] == "/camera/camera_h/color/image_raw/compressed"
        assert signals["active_eef_pose"]["topic"] == "/arm_master_r_status"
        assert signals["active_eef_pose"]["xyz"] is None
        assert signals["gripper"]["value"] is None
        assert signals["gripper"]["source"] == "field:joint_pos[6]"
    finally:
        runtime.stop_all()


def test_robot_onboard_validation_uses_latest_adapter_samples(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        robot_profile_path="configs/robot_profiles/arx_one_default.yaml",
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
        services=ServicesToggle(
            data_source=False,
            session=False,
            vsa=False,
            post_review=False,
            health_check=False,
        ),
        tuning=TuningConfig(),
        vsa=VsaConfig(),
        vlm=VlmConfig(),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)

    class _FakeOrchestrator:
        def camera_status(self, **_kwargs):
            return {
                "topic": "/camera/camera_h/color/image_raw/compressed",
                "stream_id": "cam/camera_h/color",
                "shape": [24, 32, 3],
                "host_mono_ns": time.monotonic_ns(),
            }

        def arm_status(self, **_kwargs):
            return {
                "topic": "/arm_master_r_status",
                "stream_id": "robot/arx_r/pose",
                "host_mono_ns": time.monotonic_ns(),
                "vector_len": 27,
            }

        def latest_arm_vector(self, **_kwargs):
            import numpy as np

            vec = np.zeros(27, dtype=np.float32)
            vec[6] = -1.5
            vec[21:27] = [0.5, 0.1, 0.3, 0.01, 0.02, 0.03]
            return vec

    runtime.orchestrator = _FakeOrchestrator()

    payload = runtime.robot_profile_validate("arx_one_default")

    image = payload["canonical_signals"]["primary_image"]
    assert image["present"] is True
    assert image["shape"] == [24, 32, 3]
    assert payload["streams"][0]["sample_meta"]["stream_id"] == "cam/camera_h/color"
    eef = payload["canonical_signals"]["active_eef_pose"]
    assert eef["present"] is True
    assert eef["xyz"] == [0.5, 0.1, 0.3]
    gripper = payload["canonical_signals"]["gripper"]
    assert gripper["present"] is True
    assert gripper["value"] == -1.5
    assert gripper["state"] == "closed"


def test_unified_runtime_mjpeg_uses_active_robot_profile_frame(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        robot_profile_path="configs/robot_profiles/arx_one_default.yaml",
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
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
    runtime = UnifiedRuntime(cfg)
    runtime.start()
    try:
        import numpy as np

        class _FakeOrchestrator:
            def latest_camera_frame(self, **_kwargs):
                frame = np.zeros((24, 32, 3), dtype=np.uint8)
                frame[:, :, 2] = 255
                return frame

            def stop(self):
                pass

        runtime.orchestrator = _FakeOrchestrator()
        client = TestClient(runtime.session_app)

        data = b""
        with client.stream("GET", "/mjpeg?max_frames=1") as response:
            assert response.status_code == 200
            for chunk in response.iter_bytes():
                data += chunk

        frame = _decode_first_mjpeg_frame(data)
        assert frame is not None
        assert frame[:, :, 2].mean() > 180
        assert frame[:, :, 0].mean() < 80
        assert frame[:, :, 1].mean() < 80
    finally:
        runtime.stop_all()


def test_unified_runtime_reports_rosbag_raw_artifact_status(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime, _raw_artifacts_status

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
        services=ServicesToggle(
            data_source=False,
            session=False,
            vsa=False,
            post_review=False,
            health_check=False,
        ),
        tuning=TuningConfig(),
        vsa=VsaConfig(),
        vlm=VlmConfig(),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)
    rollout_dir = tmp_path / "rollouts" / "r1"
    _write_raw_artifacts(rollout_dir)

    status = _raw_artifacts_status(rollout_dir)
    assert status["present"] is True
    assert status["raw_format"] == "rosbag2"
    assert status["bag_present"] is True
    assert runtime.vsa_status()["memory_debug_log"] == str(
        tmp_path / "logs" / "memory_debug.jsonl"
    )


def test_unified_runtime_stop_all_idempotent(tmp_path):
    """stop_all must be callable twice + before-start without raising."""
    from robolineage_app.runtime import UnifiedRuntime

    cfg = _make_config(tmp_path)
    cfg.services = ServicesToggle(
        data_source=False,
        session=True,
        vsa=False,
        health_check=False,
    )
    runtime = UnifiedRuntime(cfg)

    # Before start: no-op
    runtime.stop_all()

    runtime.start()
    runtime.stop_all()
    runtime.stop_all()  # double-stop after start: no-op


def test_unified_runtime_services_off_skip_subrunners(tmp_path):
    """services flags toggle whether session / vsa run; smoke them off."""
    from robolineage_app.runtime import UnifiedRuntime

    cfg = _make_config(tmp_path)
    cfg.services = ServicesToggle(
        data_source=False, session=False, vsa=False, health_check=False,
    )
    runtime = UnifiedRuntime(cfg)
    runtime.start()
    try:
        assert runtime.session_app is None
        assert runtime._vsa_thread is None
    finally:
        runtime.stop_all()


def test_unified_runtime_defers_data_source_without_robot_binding(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    cfg = Config(
        rollout=RolloutConfig(task_id="generic_task", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
        services=ServicesToggle(
            data_source=True,
            session=False,
            vsa=False,
            post_review=False,
            health_check=False,
        ),
        tuning=TuningConfig(),
        vsa=VsaConfig(),
        vlm=VlmConfig(),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)
    runtime.start()
    try:
        assert runtime.orchestrator is None
    finally:
        runtime.stop_all()


def test_unified_runtime_starts_each_vsa_rollout_in_new_dir(tmp_path, monkeypatch):
    """Manual rollout control should create one VSA output directory per episode."""
    from robolineage_app.runtime import UnifiedRuntime
    _install_fake_ros_topic_vsa(monkeypatch)

    task_config = tmp_path / "task_config.yaml"
    task_config.write_text(
        "\n".join(
            [
                "task_description: pick",
                "phases:",
                "  - approach",
                "  - grasp",
            ]
        ),
        encoding="utf-8",
    )
    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
        services=ServicesToggle(
            data_source=False,
            session=False,
            vsa=False,
            health_check=False,
        ),
        tuning=TuningConfig(idle_timeout=30.0),
        vsa=VsaConfig(
            camera_topic="/cam/x/image/compressed",
            arm_topic="/robot/x/state",
        ),
        adapter=_ros_topic_adapter_config(),
        vlm=VlmConfig(),        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)

    runtime.configure_vsa_task(str(task_config))
    first = runtime.start_vsa_rollout()
    runtime.stop_vsa()
    second = runtime.start_vsa_rollout()
    runtime.stop_vsa()
    assert _wait_until(lambda: not runtime.vsa_status()["analysis_draining"])

    assert first["rollout_id"] != second["rollout_id"]
    assert (tmp_path / "rollouts" / first["rollout_id"] / "task_config.yaml").exists()
    assert (tmp_path / "rollouts" / second["rollout_id"] / "task_config.yaml").exists()
    assert (tmp_path / "rollouts" / first["rollout_id"] / "raw" / "raw_manifest.json").exists()
    assert (tmp_path / "rollouts" / second["rollout_id"] / "raw" / "raw_manifest.json").exists()
    assert (tmp_path / "rollouts" / first["rollout_id"] / "task_config_binding.json").exists()
    context = json.loads((tmp_path / "rollouts" / first["rollout_id"] / "rollout_context.json").read_text())
    assert context["task_config"]["version_path"] == str(task_config)
    assert first["output_jsonl"] != second["output_jsonl"]
    debug_log = tmp_path / "logs" / "memory_debug.jsonl"
    assert first["memory_debug_log"] == str(debug_log)
    rows = [
        json.loads(line)
        for line in debug_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = {row["event"] for row in rows}
    assert {"rollout_started", "rollout_capture_stopped", "rollout_analysis_completed"} <= events
    started_rows = [row for row in rows if row["event"] == "rollout_started"]
    assert started_rows[0]["extra"]["configured_ring_capacity"] == 120
    assert started_rows[0]["extra"]["ring_capacity"] == 60


def test_unified_runtime_queues_post_review_after_vsa_stop(tmp_path, monkeypatch):
    from robolineage_app.runtime import UnifiedRuntime
    _install_fake_ros_topic_vsa(monkeypatch)

    task_config = tmp_path / "task_config.yaml"
    task_config.write_text(
        "\n".join(
            [
                "task_description: pick",
                "phases:",
                "  - approach",
                "  - grasp",
            ]
        ),
        encoding="utf-8",
    )
    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
        services=ServicesToggle(
            data_source=False,
            session=False,
            vsa=False,
            post_review=True,
            health_check=False,
        ),
        tuning=TuningConfig(idle_timeout=30.0),
        vsa=VsaConfig(
            camera_topic="/cam/x/image/compressed",
            arm_topic="/robot/x/state",
        ),
        adapter=_ros_topic_adapter_config(),
        vlm=VlmConfig(),
        post_review=PostReviewConfig(use_vlm=False, idle_delay_sec=0.0),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)
    runtime.start()
    try:
        runtime.configure_vsa_task(str(task_config))
        started = runtime.start_vsa_rollout()
        runtime.stop_vsa()

        assert _wait_until(
            lambda: runtime.post_review_status().get("last_rollout") == started["rollout_id"]
        )
        rollout_dir = tmp_path / "rollouts" / started["rollout_id"]
        assert (rollout_dir / "rollout_summary.json").exists()
        assert (rollout_dir / "dataset_admission.json").exists()
        rollouts = runtime.post_review_rollouts()
        assert rollouts["rollouts"][0]["rollout_id"] == started["rollout_id"]
        detail = runtime.post_review_detail(started["rollout_id"])
        assert detail["rollout"]["status"] == "completed"
        assert detail["dataset_admission"]["decision"] == "rejected"
    finally:
        runtime.stop_all()


def test_unified_runtime_deployment_session_runs_policy_eval_summary(tmp_path, monkeypatch):
    from robolineage_app.runtime import UnifiedRuntime
    _install_fake_ros_topic_vsa(monkeypatch)

    task_config = tmp_path / "task_config.yaml"
    task_config.write_text(
        "\n".join(
            [
                "task_description: pick",
                "phases:",
                "  - approach",
                "  - grasp",
            ]
        ),
        encoding="utf-8",
    )
    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
        services=ServicesToggle(
            data_source=False,
            session=False,
            vsa=False,
            post_review=False,
            health_check=False,
        ),
        tuning=TuningConfig(idle_timeout=30.0),
        vsa=VsaConfig(
            camera_topic="/cam/x/image/compressed",
            arm_topic="/robot/x/state",
        ),
        adapter=_ros_topic_adapter_config(),
        vlm=VlmConfig(),
        post_review=PostReviewConfig(use_vlm=False, idle_delay_sec=0.0),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)
    runtime.start()
    try:
        runtime.configure_vsa_task(str(task_config))
        session = runtime.start_deployment_session({"policy_version": "policy_1"})
        started = runtime.start_vsa_rollout()
        runtime.stop_vsa()
        stopped = runtime.stop_deployment_session()
        assert stopped["status"] == "finalizing"
        assert runtime._wait_for_rollout_group_finalizer(timeout=5.0)
        stopped = runtime.rollout_session_state()

        rollout_dir = tmp_path / "rollouts" / started["rollout_id"]
        assert session["kind"] == "deployment"
        assert (rollout_dir / "policy_evaluation.json").exists()
        assert not (rollout_dir / "dataset_admission.json").exists()
        decision = stopped["summary"]["deployment_decision"]
        assert decision["policy_version"] == "policy_1"
        assert decision["decision"] == "rollback_recommended"
        summary_dir = tmp_path / "deployment_sessions" / stopped["stopped_session"]["session_id"]
        assert (summary_dir / "policy_eval_summary.json").exists()
        assert (summary_dir / "deployment_session_report.md").exists()
        history = runtime.task_deployment_sessions(tmp_path.name)["sessions"]
        assert history[0]["session_id"] == stopped["stopped_session"]["session_id"]
        assert history[0]["decision"] == "rollback_recommended"
        detail = runtime.task_deployment_session_detail(tmp_path.name, history[0]["session_id"])
        assert detail["deployment_decision"]["decision"] == "rollback_recommended"
        assert detail["rollouts"][0]["rollout_id"] == started["rollout_id"]
    finally:
        runtime.stop_all()


def test_collection_session_stop_finalizes_until_post_review_idle(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    class _FakePostReviewWorker:
        def __init__(self) -> None:
            self.wait_calls: list[float | None] = []
            self.waiting = threading.Event()
            self.release = threading.Event()

        def status(self) -> dict:
            return {"active": True, "queue_size": 1 if self.waiting.is_set() and not self.release.is_set() else 0}

        def wait_idle(self, timeout: float | None = 10.0) -> bool:
            self.wait_calls.append(timeout)
            self.waiting.set()
            assert self.release.wait(timeout=2.0)
            return True

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
        services=ServicesToggle(
            data_source=False,
            session=False,
            vsa=False,
            post_review=False,
            health_check=False,
        ),
        tuning=TuningConfig(idle_timeout=30.0),
        vsa=VsaConfig(),
        vlm=VlmConfig(),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)
    worker = _FakePostReviewWorker()
    runtime._post_review_worker = worker
    runtime._rollout_group = {
        "kind": "collection",
        "session_id": "collection_1",
        "started_at": "2026-05-09T00:00:00Z",
        "rollout_ids": [],
    }

    stopped = runtime.stop_collection_session()

    assert stopped["status"] == "finalizing"
    assert stopped["finalizing"] is True
    assert _wait_until(lambda: worker.waiting.is_set())
    assert not (tmp_path / "collection_sessions" / "collection_1" / "collection_session_summary.json").exists()

    worker.release.set()
    assert runtime._wait_for_rollout_group_finalizer(timeout=2.0)
    state = runtime.rollout_session_state()

    assert worker.wait_calls == [None]
    assert state["status"] == "completed"
    assert state["summary"]["session_id"] == "collection_1"
    assert state["summary"]["training_ready"] is True
    assert (tmp_path / "collection_sessions" / "collection_1" / "collection_session_summary.json").exists()


def test_unified_runtime_training_framework_demo_runs_from_post_review_outputs(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    rollout = tmp_path / "rollouts" / "r1"
    rollout.mkdir(parents=True)
    (rollout / "dataset_admission.json").write_text(
        json.dumps(
            {
                "decision": "accepted",
                "data_use": ["success_trajectory"],
                "reasons": ["accepted_for_training"],
            }
        ),
        encoding="utf-8",
    )
    (rollout / "rollout_summary.json").write_text(
        json.dumps({"final_success": True, "success_confidence": 0.9}),
        encoding="utf-8",
    )
    (rollout / "failure_analysis.json").write_text(
        json.dumps({"candidate_count": 0, "failure_events": []}),
        encoding="utf-8",
    )
    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
        services=ServicesToggle(
            data_source=False,
            session=False,
            vsa=False,
            post_review=False,
            health_check=False,
        ),
        tuning=TuningConfig(idle_timeout=30.0),
        vsa=VsaConfig(),
        vlm=VlmConfig(),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)

    started = runtime.training_framework_run_demo()

    assert started["status"] == "started"
    assert _wait_until(lambda: runtime.training_framework_status()["active"] is False)
    assert runtime.training_framework_status()["last_error"] is None
    runs = runtime.training_framework_runs()["runs"]
    assert runs
    assert runs[0]["dataset_version"] == "v1"
    assert runs[0]["deploy_decision"] == "deploy_recommended"
    detail = runtime.training_framework_detail(runs[0]["run_id"])
    assert detail["deployment_recommendation"]["gating_result"] == "pass"
    assert detail["policy_context"]["dataset"]["selected_manifest_entries"] == 1


def test_unified_runtime_task_registry_and_training_selection(tmp_path, monkeypatch):
    from robolineage_app.runtime import UnifiedRuntime

    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "stack_blocks"
    rollouts = task_dir / "rollouts"
    session_dir = task_dir / "collection_sessions" / "collection_1"
    for path in (rollouts / "r1", rollouts / "r2", session_dir):
        path.mkdir(parents=True)
    (task_dir / "task_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "RoboLineage.task_manifest.v1",
                "task_id": "stack_blocks",
                "display_name": "stack blocks",
                "task_description": "stack the red block on the blue block",
            }
        ),
        encoding="utf-8",
    )
    (rollouts / "r1" / "dataset_admission.json").write_text(
        json.dumps({"decision": "accepted"}),
        encoding="utf-8",
    )
    _write_raw_artifacts(rollouts / "r1")
    (rollouts / "r2" / "dataset_admission.json").write_text(
        json.dumps(
            {
                "decision": "needs_review",
                "accepted_for_training": True,
                "label_quality": "uncertain",
                "review_reason": "labels need correction but trajectory is useful",
            }
        ),
        encoding="utf-8",
    )
    _write_raw_artifacts(rollouts / "r2")
    (session_dir / "collection_session_summary.json").write_text(
        json.dumps(
            {
                "session_id": "collection_1",
                "rollout_ids": ["r1", "r2"],
                "rollout_count": 2,
                "dataset_decision_counts": {"accepted": 1, "needs_review": 1},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ROBOLINEAGE_TASKS_ROOT", str(tasks_root))
    monkeypatch.setenv("ROBOLINEAGE_TASK_DIR", str(task_dir))

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(rollouts)),
        services=ServicesToggle(
            data_source=False,
            session=False,
            vsa=False,
            health_check=False,
        ),
        tuning=TuningConfig(),
        vsa=VsaConfig(),
        vlm=VlmConfig(),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)

    registry = runtime.task_registry()
    assert registry["active_task_id"] == "stack_blocks"
    assert registry["tasks"][0]["display_name"] == "stack blocks"
    sessions = runtime.task_collection_sessions("stack_blocks")["sessions"]
    assert sessions[0]["dataset_decision_counts"] == {"accepted": 1, "needs_review": 1}

    created = runtime.training_selection_create(
        "stack_blocks",
        {
            "source_collection_session_ids": ["collection_1"],
            "include_decisions": ["accepted", "needs_review"],
        },
    )

    selection = created["selection"]
    assert selection["rollout_ids"] == ["r1", "r2"]
    assert selection["excluded_rollouts"] == []
    assert (task_dir / "training_selections" / f"{selection['selection_id']}.json").exists()


def test_post_review_runtime_uses_4096_default_output_tokens(tmp_path, monkeypatch):
    from robolineage_app.runtime import UnifiedRuntime

    captured: dict[str, object] = {}

    class _DummyWorker:
        def __init__(self, *, agent_factory, idle_delay_sec):
            self.agent_factory = agent_factory
            self.idle_delay_sec = idle_delay_sec
            self.stop_event = threading.Event()

        def start(self):
            self.agent_factory()

    def _fake_make_vlm_runner_from_env(
        prefix,
        *,
        timeout_default,
        max_output_tokens_default,
        min_timeout_s=None,
        min_output_tokens=None,
    ):
        captured["prefix"] = prefix
        captured["timeout_default"] = timeout_default
        captured["min_timeout_s"] = min_timeout_s
        captured["max_output_tokens_default"] = max_output_tokens_default
        captured["min_output_tokens"] = min_output_tokens
        return object()

    monkeypatch.setattr(
        "robolineage_shared_agents.visual_snapshot.vlm_runner.make_vlm_runner_from_env",
        _fake_make_vlm_runner_from_env,
    )
    monkeypatch.setattr(
        "robolineage_shared_agents.visual_snapshot.vlm_priority.OfflineVLMRunner",
        lambda base_runner, *args, **kwargs: base_runner,
    )
    monkeypatch.setattr("robolineage_post_rollout.PostRolloutReviewWorker", _DummyWorker)

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(tmp_path / "rollouts")),
        services=ServicesToggle(data_source=False, session=False, vsa=False, health_check=False),
        tuning=TuningConfig(),
        vsa=VsaConfig(),
        vlm=VlmConfig(max_output_tokens=256),
        post_review=PostReviewConfig(use_vlm=True, idle_delay_sec=0.0),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)

    runtime._start_post_review()

    assert captured["prefix"] == "POST_REVIEW_VLM"
    assert captured["timeout_default"] == 60.0
    assert captured["min_timeout_s"] == 60.0
    assert captured["max_output_tokens_default"] == 4096
    assert captured["min_output_tokens"] == 4096


def test_task_registry_ignores_console_scratch_dir_until_task_is_created(tmp_path, monkeypatch):
    from robolineage_app.runtime import UnifiedRuntime

    tasks_root = tmp_path / "tasks"
    scratch_dir = tmp_path / ".runtime" / "console"
    (scratch_dir / "rollouts").mkdir(parents=True)
    monkeypatch.setenv("ROBOLINEAGE_TASKS_ROOT", str(tasks_root))
    monkeypatch.setenv("ROBOLINEAGE_TASK_DIR", str(scratch_dir))

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(scratch_dir / "rollouts")),
        services=ServicesToggle(
            data_source=False,
            session=False,
            vsa=False,
            health_check=False,
        ),
        tuning=TuningConfig(),
        vsa=VsaConfig(),
        vlm=VlmConfig(),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)

    registry = runtime.task_registry()
    assert registry["root"] == str(tasks_root)
    assert registry["active_task_id"] is None
    assert registry["active_task_dir"] is None
    assert registry["tasks"] == []

    created = runtime.task_create({"task_description": "stack the red block"})
    assert created["task"]["task_description"] == "stack the red block"
    registry = runtime.task_registry()
    assert registry["active_task_id"] == created["task"]["task_id"]
    assert [task["task_id"] for task in registry["tasks"]] == [created["task"]["task_id"]]


def test_training_run_resolves_only_task_scoped_ids(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    task_dir = tmp_path / "tasks" / "stack_blocks"
    task_dir.mkdir(parents=True)
    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(task_dir / "rollouts")),
        services=ServicesToggle(
            data_source=False,
            session=False,
            vsa=False,
            health_check=False,
        ),
        tuning=TuningConfig(),
        vsa=VsaConfig(),
        vlm=VlmConfig(),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)

    with pytest.raises(ValueError, match="framework_profile_path"):
        runtime._resolve_profile_path(
            task_dir,
            {"framework_profile_path": str(tmp_path / "outside.yaml")},
        )
    with pytest.raises(ValueError, match="selection_path"):
        runtime._resolve_selection_path(
            task_dir,
            {"selection_path": str(tmp_path / "outside.json")},
        )
    with pytest.raises(FileNotFoundError):
        runtime._resolve_profile_path(task_dir, {"framework_profile_id": "../outside"})


def test_selected_adapted_dataset_can_reuse_compatible_new_profile(tmp_path):
    from robolineage_app.runtime import UnifiedRuntime

    task_dir = tmp_path / "tasks" / "stack_blocks"
    task_dir.mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    selection_path = task_dir / "training_selections" / "sel_1.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(
        json.dumps({"selection_id": "sel_1", "rollout_ids": ["r1"], "include_decisions": ["accepted"]}),
        encoding="utf-8",
    )
    old_profile = _write_runtime_training_profile(task_dir, repo, "profile_old", train_label="old")
    new_profile = _write_runtime_training_profile(task_dir, repo, "profile_new", train_label="new")
    run_dir = task_dir / "training_runs" / "run_1"
    (run_dir / "framework").mkdir(parents=True)
    (run_dir / "framework" / "dataset_adapt_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "adapter_id": "generated_act_hdf5",
                "adapter_strategy": "generated_repo_specific_script",
                "target_dataset_kind": "act_hdf5",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training_run_config.json").write_text(
        json.dumps(
            {
                "policy_version": "old_policy",
                "selection_path": str(selection_path),
                "framework_profile_path": str(old_profile),
            }
        ),
        encoding="utf-8",
    )
    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="op"),
        recorder=RecorderConfig(output_dir=str(task_dir / "rollouts")),
        services=ServicesToggle(data_source=False, session=False, vsa=False, health_check=False),
        tuning=TuningConfig(),
        vsa=VsaConfig(),
        vlm=VlmConfig(),
        health=HealthConfig(port=8081),
    )
    runtime = UnifiedRuntime(cfg)

    assert runtime._find_adapted_training_run(
        task_dir=task_dir,
        selection_path=selection_path,
        profile_path=new_profile,
        policy_version="new_policy",
        requested_run_id="run_1",
    ) == run_dir


def _write_runtime_training_profile(task_dir: Path, repo: Path, profile_id: str, *, train_label: str) -> Path:
    profile_dir = task_dir / "framework_profiles" / profile_id
    profile_dir.mkdir(parents=True)
    path = profile_dir / "framework_profile.generated.yaml"
    path.write_text(
        "\n".join(
            [
                f"name: {profile_id}",
                "framework_type: generic",
                f"repo_root: {repo}",
                "train_command:",
                f"  args: [bash, -lc, 'echo {train_label} {{dataset_output}}']",
                "outputs:",
                "  checkpoint_glob: '{checkpoint_dir}/*.ckpt'",
                "dataset_adapter:",
                "  adapter_id: generated_act_hdf5",
                "  strategy: generated_repo_specific_script",
                "  target_contract:",
                "    dataset_kind: act_hdf5",
                "    camera_names: [head, left_wrist, right_wrist]",
                "    fields:",
                "      - path: /action",
                "        role: action",
                "        required: true",
                "        shape: '(T, 14)'",
            ]
        ),
        encoding="utf-8",
    )
    return path
