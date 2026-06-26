from __future__ import annotations

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


def test_unified_runtime_robot_onboarding_generates_profile_and_validation(tmp_path: Path, monkeypatch) -> None:
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
        payload = response.json()
        assert payload["status"] == "generated"
        assert payload["robot_id"] == "arx_one_default"
        assert payload["validation"]["status"] == "ok"
        assert payload["report"]["validation_status"] == "ok"
        assert Path(payload["generated_profile_path"]).parent == profiles_root
        assert (task_root / "robot_onboarding" / payload["job_id"] / "robot_onboarding_report.json").exists()

        robots = client.get("/robots").json()
        ids = [item["robot_id"] for item in robots["profiles"]]
        assert "arx_one_default" in ids
    finally:
        runtime.stop_all()
