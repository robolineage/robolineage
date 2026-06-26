from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TopicInfo:
    name: str
    types: tuple[str, ...]


def build_topic_probe_report(topics: Iterable[TopicInfo | tuple[str, Iterable[str]]]) -> dict[str, Any]:
    """Return profile-building hints from a ROS2 topic inventory.

    The probe is intentionally advisory. It helps an operator or Robot
    Onboarding Agent choose candidate streams, while the final binding still
    lives in a reviewed robot profile.
    """

    normalized = [_normalize_topic(item) for item in topics]
    cameras = [_candidate(item, role="camera") for item in normalized if _is_camera_topic(item)]
    states = [_candidate(item, role="robot_state") for item in normalized if _is_state_topic(item)]
    controls = [_candidate(item, role="action_or_control") for item in normalized if _is_control_topic(item)]
    return {
        "schema_version": "RoboLineage.topic_probe.v1",
        "topic_count": len(normalized),
        "camera_candidates": cameras,
        "robot_state_candidates": states,
        "action_or_control_candidates": controls,
        "profile_hint": {
            "active_color_image": cameras[0]["name"] if cameras else None,
            "active_robot_state": states[0]["name"] if states else None,
        },
        "notes": [
            "Topic probing is advisory; review the generated robot profile before collection.",
            "Prefer compressed color image topics and end-effector or joint-state topics with stable timestamps.",
        ],
    }


def probe_ros2_topics(*, ros_domain_id: int | None = None) -> dict[str, Any]:
    """Probe the active ROS2 graph and return onboarding hints.

    This function imports ROS2 packages lazily so the repository remains
    importable on non-ROS development machines.
    """

    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:  # pragma: no cover - exercised only without ROS2.
        raise RuntimeError("ROS2 Python packages are required for topic probing") from exc

    owns_context = not _rclpy_context_ok(rclpy)
    if owns_context:
        rclpy.init(args=None, domain_id=ros_domain_id)
    node = Node("robolineage_topic_probe")
    try:
        topics = [
            TopicInfo(name=name, types=tuple(types))
            for name, types in node.get_topic_names_and_types()
        ]
        return build_topic_probe_report(topics)
    finally:
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()


def _normalize_topic(item: TopicInfo | tuple[str, Iterable[str]]) -> TopicInfo:
    if isinstance(item, TopicInfo):
        return item
    name, types = item
    return TopicInfo(name=str(name), types=tuple(str(value) for value in types))


def _candidate(topic: TopicInfo, *, role: str) -> dict[str, Any]:
    return {
        "name": topic.name,
        "types": list(topic.types),
        "role": role,
        "score": _score(topic, role=role),
    }


def _is_camera_topic(topic: TopicInfo) -> bool:
    text = _topic_text(topic)
    return (
        "sensor_msgs/msg/compressedimage" in text
        or "sensor_msgs/msg/image" in text
        or ("camera" in text and any(token in text for token in ("image", "color", "rgb")))
    )


def _is_state_topic(topic: TopicInfo) -> bool:
    text = _topic_text(topic)
    if any(token in text for token in ("cmd", "command", "trajectory", "goal")):
        return False
    return any(
        token in text
        for token in (
            "jointstate",
            "robotstate",
            "robot_status",
            "eef",
            "end_effector",
            "pose",
            "arm",
            "gripper",
        )
    )


def _is_control_topic(topic: TopicInfo) -> bool:
    text = _topic_text(topic)
    return any(token in text for token in ("cmd", "command", "trajectory", "action", "servo"))


def _score(topic: TopicInfo, *, role: str) -> float:
    text = _topic_text(topic)
    score = 0.5
    if role == "camera":
        if "compressedimage" in text:
            score += 0.25
        if "color" in text or "rgb" in text:
            score += 0.15
        if "depth" in text:
            score -= 0.2
    elif role == "robot_state":
        if "eef" in text or "end_effector" in text:
            score += 0.2
        if "jointstate" in text or "robotstate" in text or "robot_status" in text:
            score += 0.15
        if "gripper" in text:
            score += 0.1
    return round(max(0.0, min(1.0, score)), 3)


def _topic_text(topic: TopicInfo) -> str:
    return " ".join([topic.name, *topic.types]).replace("/", " ").replace("_", " ").lower()


def _rclpy_context_ok(rclpy: Any) -> bool:
    ok = getattr(rclpy, "ok", None)
    if not callable(ok):
        return False
    try:
        return bool(ok())
    except TypeError:
        return bool(ok)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Probe ROS2 topics and print RoboLineage onboarding hints.")
    parser.add_argument("--ros-domain-id", type=int, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(probe_ros2_topics(ros_domain_id=args.ros_domain_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised on ROS2 workstations.
    raise SystemExit(main())
