from __future__ import annotations

from pathlib import Path

from robolineage_robot import load_robot_profile, profile_to_adapter_config, profile_to_vsa_topics


def test_example_profiles_are_not_loaded_by_default(monkeypatch) -> None:
    from robolineage_robot.profile import default_profile_roots

    monkeypatch.delenv("ROBOLINEAGE_ROBOT_PROFILES_ROOT", raising=False)
    monkeypatch.delenv("ROBOLINEAGE_INCLUDE_EXAMPLE_ROBOT_PROFILES", raising=False)

    roots = [str(path) for path in default_profile_roots()]

    assert "configs/robot_profiles" not in " ".join(roots)


def test_generic_profile_has_no_robot_family_leakage(tmp_path: Path) -> None:
    profile_path = tmp_path / "generic_bot.yaml"
    profile_path.write_text(
        """
schema_version: RoboLineage.robot_profile.v1
robot_id: generic_bot
display_name: Generic bot
connection:
  type: ros2
  ros_domain_id: 17
streams:
  color_images:
    primary:
      topic: /camera/color
      msg_type: sensor_msgs/msg/CompressedImage
      transport: compressed
      canonical_topic: camera/primary/color
  robot_states:
    active_arm:
      topic: /robot/state
      msg_type: example_msgs/msg/RobotState
      canonical_state_topic: robot/active/state
      eef_position_field: tool.position
      eef_orientation_field: tool.rotation
      gripper_field: gripper.position
active_streams:
  color_image: primary
  robot_state: active_arm
ROBOLINEAGE_bindings:
  vsa:
    canonical_camera_topic: camera/primary/color
    canonical_arm_topic: robot/active/state
""",
        encoding="utf-8",
    )

    profile = load_robot_profile(profile_path)
    adapter = profile_to_adapter_config(profile)
    camera_topic, arm_topic = profile_to_vsa_topics(profile)
    payload = str({
        "summary": {
            key: value
            for key, value in profile.to_summary(active=True).items()
            if key != "profile_path"
        },
        "adapter_type": adapter.type,
        "camera_topic": camera_topic,
        "arm_topic": arm_topic,
        "camera_stream": adapter.cameras["primary"].stream_id,
        "arm_stream": adapter.arms["active_arm"].state_stream_id,
    }).lower()

    assert adapter.type == "ros2_profile"
    assert camera_topic == "/camera/color"
    assert arm_topic == "/robot/state"
    assert "arx" not in payload


def test_arx_profile_maps_to_current_runtime_topics() -> None:
    profile = load_robot_profile(Path("configs/robot_profiles/arx_one_default.yaml"))

    adapter = profile_to_adapter_config(profile)
    camera_topic, arm_topic = profile_to_vsa_topics(profile)

    assert profile.robot_id == "arx_one_default"
    assert adapter.type == "ros2_profile"
    assert adapter.ros_domain_id == 195
    assert set(adapter.cameras) == {"camera_h", "camera_l", "camera_r"}
    assert adapter.cameras["camera_h"].stream_id == "cam/camera_h/color"
    assert adapter.cameras["camera_l"].stream_id == "cam/camera_l/color"
    assert adapter.cameras["camera_r"].camera_name == "camera_r"
    assert adapter.arms["right_arm"].slave_status == "/arm_master_r_status"
    assert adapter.arms["right_arm"].gripper_field == "joint_pos[6]"
    assert profile.to_summary()["gripper_source"] == "field:joint_pos[6]"
    assert camera_topic == "/camera/camera_h/color/image_raw/compressed"
    assert arm_topic == "/arm_master_r_status"


def test_sanitized_multi_robot_profiles_load_and_bind() -> None:
    for name in ["realman_default.yaml", "galbot_g1_default.yaml"]:
        profile = load_robot_profile(Path("configs/robot_profiles") / name)
        adapter = profile_to_adapter_config(profile)
        camera_topic, arm_topic = profile_to_vsa_topics(profile)

        assert profile.connection_type == "ros2"
        assert adapter.type == "ros2_profile"
        assert camera_topic and camera_topic.startswith("/")
        assert arm_topic and arm_topic.startswith("/")
        assert profile.to_summary()["read_only"] is True
