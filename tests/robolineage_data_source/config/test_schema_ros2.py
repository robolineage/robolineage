"""Phase 1 ROS2 adapter config schema and loader coverage."""
from __future__ import annotations

import textwrap
from pathlib import Path

from robolineage_data_source.config.loader import load_config
from robolineage_data_source.config.schema import (
    ArmTopicSpec,
    CameraTopicSpec,
    Config,
    RolloutConfig,
    Ros2AdapterConfig,
)


def test_ros2_adapter_config_defaults():
    cfg = Ros2AdapterConfig()
    assert cfg.type == "ros2_profile"
    assert cfg.ros_domain_id == 0
    assert cfg.spin_threads == 2
    assert cfg.cameras == {}
    assert cfg.arms == {}
    assert cfg.master_overlay_topic is None
    assert cfg.enable_depth is False


def test_camera_topic_spec_required_fields():
    spec = CameraTopicSpec(
        topic="/camera/camera_h/color/image_raw/compressed",
        transport="compressed",
        qos="sensor_data",
        stream_id="cam/camera_h/color",
    )
    assert spec.transport == "compressed"
    assert spec.camera_name is None


def test_arm_topic_spec_optional_master_command():
    spec = ArmTopicSpec(
        slave_status="/arm_slave_l_status",
        state_stream_id="robot/left_arm/state",
        qos="reliable",
    )
    assert spec.master_command is None


def test_config_adapter_optional():
    cfg = Config(rollout=RolloutConfig(task_id="t", operator_id="o"))
    assert cfg.adapter is None


def test_load_config_with_adapter_section(tmp_path: Path):
    yaml_text = textwrap.dedent(
        """
        rollout:
          task_id: t
          operator_id: o
        adapter:
          type: ros2_profile
          ros_domain_id: 0
          spin_threads: 2
          master_overlay_topic: robot/active/command
        cameras_ros2:
          camera_h:
            topic: /camera/camera_h/color/image_raw/compressed
            transport: compressed
            qos: sensor_data
            stream_id: cam/camera_h/color
            camera_name: camera_h
        arms_ros2:
          left_arm:
            slave_status: /arm_slave_l_status
            state_stream_id: robot/left_arm/state
            qos: reliable
            master_command: /arm_master_l_status
            msg_type: example_msgs/msg/RobotState
        """
    )
    p = tmp_path / "ros2.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.adapter is not None
    assert cfg.adapter.type == "ros2_profile"
    assert cfg.adapter.cameras["camera_h"].topic == (
        "/camera/camera_h/color/image_raw/compressed"
    )
    assert cfg.adapter.cameras["camera_h"].camera_name == "camera_h"
    assert cfg.adapter.arms["left_arm"].master_command == "/arm_master_l_status"
    assert cfg.adapter.arms["left_arm"].msg_type == "example_msgs/msg/RobotState"
    assert cfg.adapter.master_overlay_topic == "robot/active/command"


def test_load_config_without_adapter_backwards_compat(tmp_path: Path):
    yaml_text = textwrap.dedent(
        """
        rollout:
          task_id: t
          operator_id: o
        cameras:
          cam0:
            type: mock
        """
    )
    p = tmp_path / "legacy.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.adapter is None
    assert "cam0" in cfg.cameras


def test_load_config_resolves_robot_profile_path(tmp_path: Path):
    profile_dir = tmp_path / "robot_profiles"
    profile_dir.mkdir()
    profile = profile_dir / "generic.yaml"
    profile.write_text("schema_version: RoboLineage.robot_profile.v1\nrobot_id: generic\n", encoding="utf-8")
    p = tmp_path / "config.yaml"
    p.write_text(
        textwrap.dedent(
            """
            rollout:
              task_id: t
              operator_id: o
            robot_profile_path: robot_profiles/generic.yaml
            """
        ),
        encoding="utf-8",
    )

    cfg = load_config(p)

    assert cfg.robot_profile_path == str(profile.resolve())


# ---------------------------------------------------------------------------
# Phase 5 — services / tuning / vsa / vlm / health sections
# ---------------------------------------------------------------------------


def test_phase5_dataclass_defaults_match_current_runtime_defaults():
    """Defaults of runtime config dataclasses should match current app defaults."""
    from robolineage_data_source.config.schema import (
        HealthConfig,
        PostReviewConfig,
        ServicesToggle,
        TuningConfig,
        VlmConfig,
        VsaConfig,
    )

    services = ServicesToggle()
    assert services.data_source is True
    assert services.session is True
    assert services.vsa is True
    assert services.post_review is True
    assert services.health_check is True

    tuning = TuningConfig()
    assert tuning.ring_capacity == 120
    assert tuning.still_threshold == 3e-4
    assert tuning.still_min_frames == 25
    assert tuning.gripper_close_threshold == -1.0
    assert tuning.rotation_weight == 0.2
    assert tuning.smoothing_window == 10
    assert tuning.motion_resume_threshold == 8e-4
    assert tuning.context_frames == 15
    assert tuning.max_keyframes == 3
    assert tuning.idle_timeout == 10.0
    assert tuning.periodic_interval_sec == 2.0
    assert tuning.heartbeat_interval == 5.0
    assert tuning.merge_window_sec == 1.0
    assert tuning.final_settle_sec == 1.0
    assert tuning.max_vlm_windows_per_rollout is None
    assert tuning.vlm_workers == 1
    assert tuning.strong_prior_margin == 0.35
    assert tuning.prior_sticky_frames == 2

    vsa = VsaConfig()
    assert vsa.camera_topic == "camera/primary/color"
    assert vsa.arm_topic == "robot/active/state"
    assert vsa.task_config_path is None
    assert vsa.output_jsonl_path is None
    assert vsa.max_events is None

    vlm = VlmConfig()
    assert vlm.timeout == 20.0
    assert vlm.max_output_tokens == 256

    post_review = PostReviewConfig()
    assert post_review.use_vlm is True
    assert post_review.idle_delay_sec == 5.0
    assert post_review.max_review_images == 12

    health = HealthConfig()
    assert health.port == 8081
    assert health.bind == "0.0.0.0"


def test_load_config_parses_phase5_sections(tmp_path: Path):
    yaml_text = textwrap.dedent(
        """
        rollout:
          task_id: t
          operator_id: o
        services:
          data_source: true
          session: true
          vsa: false
          post_review: false
          health_check: true
        tuning:
          ring_capacity: 240
          still_threshold: 0.001
          gripper_close_threshold: -0.8
          context_frames: 20
          periodic_interval_sec: 2.0
          merge_window_sec: 0.4
          final_settle_sec: 0.8
          max_vlm_windows_per_rollout: 64
          vlm_workers: 2
        vsa:
          camera_topic: cam/camera_h/color
          arm_topic: robot/left_arm/state
          task_config_path: /etc/robolineage/task_pickplace.yaml
          output_jsonl_path: /var/log/RoboLineage/snapshots.jsonl
          max_events: 100
        vlm:
          timeout: 60.0
          max_output_tokens: 512
        post_review:
          use_vlm: false
          idle_delay_sec: 1.5
          max_review_images: 4
        health:
          port: 9090
          bind: 127.0.0.1
        """
    )
    p = tmp_path / "phase5.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p)

    assert cfg.services is not None
    assert cfg.services.vsa is False
    assert cfg.services.session is True
    assert cfg.services.post_review is False

    assert cfg.tuning is not None
    assert cfg.tuning.ring_capacity == 240
    assert cfg.tuning.still_threshold == 0.001
    assert cfg.tuning.gripper_close_threshold == -0.8
    assert cfg.tuning.context_frames == 20
    assert cfg.tuning.periodic_interval_sec == 2.0
    assert cfg.tuning.merge_window_sec == 0.4
    assert cfg.tuning.final_settle_sec == 0.8
    assert cfg.tuning.max_vlm_windows_per_rollout == 64
    assert cfg.tuning.vlm_workers == 2
    assert cfg.tuning.still_min_frames == 25
    # Unspecified fields fall back to defaults
    assert cfg.tuning.max_keyframes == 3

    assert cfg.vsa is not None
    assert cfg.vsa.camera_topic == "cam/camera_h/color"
    assert cfg.vsa.task_config_path == "/etc/robolineage/task_pickplace.yaml"
    assert cfg.vsa.max_events == 100

    assert cfg.vlm is not None
    assert cfg.vlm.timeout == 60.0
    assert cfg.vlm.max_output_tokens == 512

    assert cfg.post_review is not None
    assert cfg.post_review.use_vlm is False
    assert cfg.post_review.idle_delay_sec == 1.5
    assert cfg.post_review.max_review_images == 4

    assert cfg.health is not None
    assert cfg.health.port == 9090
    assert cfg.health.bind == "127.0.0.1"


def test_load_config_phase5_sections_optional_no_change_to_legacy(tmp_path: Path):
    """Yaml without any of the 5 new sections returns Config with all fields
    None — back-compat for older minimal runtime configs."""
    yaml_text = textwrap.dedent(
        """
        rollout:
          task_id: t
          operator_id: o
        adapter:
          type: ros2_profile
        """
    )
    p = tmp_path / "no_phase5.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.services is None
    assert cfg.tuning is None
    assert cfg.vsa is None
    assert cfg.vlm is None
    assert cfg.post_review is None
    assert cfg.health is None
