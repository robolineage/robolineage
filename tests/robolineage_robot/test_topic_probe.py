from __future__ import annotations

from robolineage_robot.topic_probe import TopicInfo, build_topic_probe_report


def test_topic_probe_ranks_camera_and_state_candidates() -> None:
    report = build_topic_probe_report([
        TopicInfo("/robot/cmd_vel", ("geometry_msgs/msg/Twist",)),
        TopicInfo("/camera/head/color/image_raw/compressed", ("sensor_msgs/msg/CompressedImage",)),
        TopicInfo("/robot/right_arm/state", ("example_msgs/msg/ManipulatorState",)),
        TopicInfo("/camera/depth/image_raw", ("sensor_msgs/msg/Image",)),
    ])

    assert report["schema_version"] == "RoboLineage.topic_probe.v1"
    assert report["topic_count"] == 4
    assert report["camera_candidates"][0]["name"] == "/camera/head/color/image_raw/compressed"
    assert report["robot_state_candidates"][0]["name"] == "/robot/right_arm/state"
    assert report["action_or_control_candidates"][0]["name"] == "/robot/cmd_vel"
    assert report["profile_hint"] == {
        "active_color_image": "/camera/head/color/image_raw/compressed",
        "active_robot_state": "/robot/right_arm/state",
    }
