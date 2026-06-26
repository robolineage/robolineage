from __future__ import annotations

import json
from pathlib import Path

import yaml

from robolineage_robot import load_robot_profile
from robolineage_robot.onboarding import RobotOnboardingAgent


class _FakeRobotOnboardingLLM:
    model = "fake-robot-onboarding-llm"

    def generate(self, context: dict) -> dict:
        assert context["normalized_profile"]["robot_id"] == "arx_one_default"
        assert context["robot_note"] == "ARX default pasted by operator"
        return {
            "profile_summary": "ARX profile with head and wrist cameras.",
            "binding_explanation": "camera_h is active for online VSA; recorder keeps camera_h and camera_r.",
            "warnings": ["confirm wrist camera latency before long collection"],
            "assumptions": ["operator pasted an existing profile"],
            "recommended_checks": ["validate active camera topic before rollout"],
        }


def test_robot_onboarding_agent_writes_generated_profile_report_and_events(tmp_path: Path) -> None:
    source_yaml = Path("configs/robot_profiles/arx_one_default.yaml").read_text(encoding="utf-8")
    task_root = tmp_path / "task_pick"
    generated_root = tmp_path / "robot_profiles"

    result = RobotOnboardingAgent(enable_env_llm=False).run(
        profile_yaml=source_yaml,
        task_root=task_root,
        generated_profiles_root=generated_root,
        robot_note="ARX default pasted by operator",
        job_id="onboard_test",
    )

    assert result.status == "generated"
    assert result.robot_id == "arx_one_default"
    assert result.generated_profile_path == generated_root / "generated_arx_one_default.yaml"
    assert result.artifact_profile_path == task_root / "robot_onboarding" / "onboard_test" / "robot_profile.generated.yaml"
    assert result.report_path.exists()
    assert result.events_path.exists()
    assert result.understanding_path.exists()

    generated = load_robot_profile(result.generated_profile_path)
    assert generated.active_color_stream_id == "camera_h"
    assert generated.active_robot_state_id == "right_arm"

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "RoboLineage.robot_onboarding_report.v1"
    assert report["robot_id"] == "arx_one_default"
    assert report["active_camera"] == "camera_h"
    assert report["active_robot_state"] == "right_arm"
    assert report["recorder_cameras"] == ["camera_h", "camera_r"]
    assert report["vsa_topics"]["camera"] == "/camera/camera_h/color/image_raw/compressed"
    assert report["llm_understanding"]["status"] == "not_configured"
    assert report["llm_understanding"]["path"] == str(result.understanding_path)

    understanding = json.loads(result.understanding_path.read_text(encoding="utf-8"))
    assert understanding["status"] == "not_configured"

    events = [json.loads(line) for line in result.events_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == [
        "onboarding_started",
        "profile_loaded",
        "schema_understood",
        "streams_identified",
        "bindings_identified",
        "recorder_policy_identified",
        "llm_understanding_not_configured",
        "understanding_written",
        "profile_written",
        "onboarding_completed",
    ]


def test_robot_onboarding_agent_adds_minimum_defaults(tmp_path: Path) -> None:
    minimal_yaml = yaml.safe_dump({
        "schema_version": "RoboLineage.robot_profile.v1",
        "robot_id": "mini_bot",
        "streams": {
            "color_images": {
                "camera_h": {
                    "topic": "/camera/color/compressed",
                    "msg_type": "sensor_msgs/msg/CompressedImage",
                    "stream_id": "cam/camera_h/color",
                },
            },
            "robot_states": {
                "right_arm": {
                    "topic": "/arm/status",
                    "msg_type": "arx5_arm_msg/msg/RobotStatus",
                    "state_stream_id": "robot/arx_r/pose",
                    "decoder": "arx5_robot_status_27_vec",
                },
            },
        },
    }, sort_keys=False)

    result = RobotOnboardingAgent(enable_env_llm=False).run(
        profile_yaml=minimal_yaml,
        task_root=tmp_path / "task_mini",
        generated_profiles_root=tmp_path / "profiles",
        job_id="onboard_minimal",
    )

    payload = yaml.safe_load(result.generated_profile_path.read_text(encoding="utf-8"))
    assert payload["display_name"] == "mini_bot"
    assert payload["active_streams"] == {"color_image": "camera_h", "robot_state": "right_arm"}
    assert payload["ROBOLINEAGE_bindings"]["vsa"]["canonical_camera_topic"] == "cam/camera_h/color"
    assert payload["ROBOLINEAGE_bindings"]["vsa"]["canonical_arm_topic"] == "robot/arx_r/pose"
    assert payload["ROBOLINEAGE_bindings"]["recorder"]["camera_names"] == ["camera_h"]
    assert payload["capabilities"]["read_only"] is True


def test_robot_onboarding_agent_writes_llm_understanding_when_configured(tmp_path: Path) -> None:
    source_yaml = Path("configs/robot_profiles/arx_one_default.yaml").read_text(encoding="utf-8")
    result = RobotOnboardingAgent(llm_client=_FakeRobotOnboardingLLM()).run(
        profile_yaml=source_yaml,
        task_root=tmp_path / "task_pick",
        generated_profiles_root=tmp_path / "robot_profiles",
        robot_note="ARX default pasted by operator",
        job_id="onboard_llm",
    )

    understanding = json.loads(result.understanding_path.read_text(encoding="utf-8"))
    assert understanding["schema_version"] == "RoboLineage.robot_onboarding_understanding.v1"
    assert understanding["status"] == "generated"
    assert understanding["model"] == "fake-robot-onboarding-llm"
    assert understanding["profile_summary"] == "ARX profile with head and wrist cameras."
    assert understanding["recommended_checks"] == ["validate active camera topic before rollout"]

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["llm_understanding"]["status"] == "generated"
    assert report["llm_understanding"]["path"] == str(result.understanding_path)
    assert report["llm_understanding"]["summary"] == "ARX profile with head and wrist cameras."

    events = [json.loads(line)["event"] for line in result.events_path.read_text(encoding="utf-8").splitlines()]
    assert "llm_understanding_started" in events
    assert "llm_understanding_completed" in events
