from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robolineage_data_source.config.schema import (
    ArmTopicSpec,
    CameraTopicSpec,
    Ros2AdapterConfig,
)


ROBOT_PROFILE_SCHEMA_VERSION = "RoboLineage.robot_profile.v1"


@dataclass(frozen=True)
class RobotProfile:
    """Minimal robot-specific information required by RoboLineage.

    A profile is intentionally smaller than a robot capability model. It only
    describes how to read the canonical streams used by collection, VSA,
    post-review and evaluation.
    """

    robot_id: str
    display_name: str
    path: Path
    payload: dict[str, Any]

    @property
    def schema_version(self) -> str:
        return str(self.payload.get("schema_version") or "")

    @property
    def connection_type(self) -> str:
        connection = self.payload.get("connection")
        if isinstance(connection, dict):
            return str(connection.get("type") or "")
        return ""

    @property
    def ros_domain_id(self) -> int | None:
        connection = self.payload.get("connection")
        if isinstance(connection, dict) and connection.get("ros_domain_id") is not None:
            return int(connection["ros_domain_id"])
        return None

    @property
    def namespace(self) -> str:
        connection = self.payload.get("connection")
        if isinstance(connection, dict):
            return str(connection.get("namespace") or "")
        return ""

    @property
    def active_color_stream_id(self) -> str | None:
        active = self.payload.get("active_streams")
        if isinstance(active, dict) and active.get("color_image"):
            return str(active["color_image"])
        color_images = self.color_images
        return next(iter(color_images), None)

    @property
    def active_robot_state_id(self) -> str | None:
        active = self.payload.get("active_streams")
        if isinstance(active, dict) and active.get("robot_state"):
            return str(active["robot_state"])
        states = self.robot_states
        return next(iter(states), None)

    @property
    def color_images(self) -> dict[str, dict[str, Any]]:
        streams = self.payload.get("streams")
        if not isinstance(streams, dict):
            return {}
        value = streams.get("color_images")
        return value if isinstance(value, dict) else {}

    @property
    def robot_states(self) -> dict[str, dict[str, Any]]:
        streams = self.payload.get("streams")
        if not isinstance(streams, dict):
            return {}
        value = streams.get("robot_states")
        return value if isinstance(value, dict) else {}

    def color_stream(self, stream_id: str | None = None) -> dict[str, Any] | None:
        key = stream_id or self.active_color_stream_id
        if key is None:
            return None
        value = self.color_images.get(key)
        return value if isinstance(value, dict) else None

    def robot_state_stream(self, stream_id: str | None = None) -> dict[str, Any] | None:
        key = stream_id or self.active_robot_state_id
        if key is None:
            return None
        value = self.robot_states.get(key)
        return value if isinstance(value, dict) else None

    def to_summary(self, *, active: bool = False) -> dict[str, Any]:
        color = self.color_stream() or {}
        state = self.robot_state_stream() or {}
        gripper_source = _gripper_source_label(state)
        return {
            "robot_id": self.robot_id,
            "display_name": self.display_name,
            "profile_path": str(self.path),
            "schema_version": self.schema_version,
            "connection_type": self.connection_type,
            "ros_domain_id": self.ros_domain_id,
            "namespace": self.namespace,
            "active": active,
            "active_color_stream": self.active_color_stream_id,
            "active_robot_state": self.active_robot_state_id,
            "color_topic": color.get("topic"),
            "color_msg_type": color.get("msg_type"),
            "state_topic": state.get("topic"),
            "state_msg_type": state.get("msg_type"),
            "gripper_source": gripper_source,
            "gripper_field": state.get("gripper_field"),
            "read_only": _capability_flag(self.payload, "read_only", default=True),
            "policy_drive": _capability_flag(self.payload, "policy_drive", default=False),
        }


class RobotProfileRegistry:
    def __init__(self, roots: list[Path] | None = None) -> None:
        self.roots = roots or default_profile_roots()

    def list_profiles(self) -> list[RobotProfile]:
        profiles: list[RobotProfile] = []
        seen: set[str] = set()
        for root in self.roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")):
                try:
                    profile = load_robot_profile(path)
                except Exception:
                    continue
                if profile.robot_id in seen:
                    continue
                profiles.append(profile)
                seen.add(profile.robot_id)
        return profiles

    def get(self, robot_id: str) -> RobotProfile:
        for profile in self.list_profiles():
            if profile.robot_id == robot_id:
                return profile
        raise FileNotFoundError(f"robot profile not found: {robot_id}")


def default_profile_roots() -> list[Path]:
    roots: list[Path] = []
    import os

    env_root = os.environ.get("ROBOLINEAGE_ROBOT_PROFILES_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.append(Path.cwd() / "robot_profiles")
    if os.environ.get("ROBOLINEAGE_INCLUDE_EXAMPLE_ROBOT_PROFILES") == "1":
        roots.append(Path.cwd() / "configs" / "robot_profiles")
    return roots


def load_robot_profile(path: str | Path) -> RobotProfile:
    profile_path = Path(path).expanduser()
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"robot profile must be a mapping: {profile_path}")
    if str(payload.get("schema_version") or "") != ROBOT_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"unsupported robot profile schema: {payload.get('schema_version')!r}")
    robot_id = str(payload.get("robot_id") or profile_path.stem).strip()
    if not robot_id:
        raise ValueError("robot profile missing robot_id")
    display_name = str(payload.get("display_name") or robot_id)
    return RobotProfile(
        robot_id=robot_id,
        display_name=display_name,
        path=profile_path,
        payload=payload,
    )


def profile_to_adapter_config(profile: RobotProfile) -> Ros2AdapterConfig:
    """Create the current compatible ROS2 adapter config from a profile.

    This keeps today's deployed platform working while the runtime moves
    toward profile-driven robot onboarding.
    """

    connection = profile.payload.get("connection") if isinstance(profile.payload.get("connection"), dict) else {}
    cameras = {
        name: CameraTopicSpec(
            topic=str(spec["topic"]),
            transport=str(spec.get("transport") or "compressed"),
            qos=str(spec.get("qos") or "sensor_data"),
            stream_id=str(spec.get("stream_id") or spec.get("canonical_topic") or f"camera/{name}/color"),
            camera_name=name,
        )
        for name, spec in profile.color_images.items()
        if isinstance(spec, dict) and spec.get("topic")
    }
    arms = {
        name: ArmTopicSpec(
            slave_status=str(spec["topic"]),
            state_stream_id=str(spec.get("state_stream_id") or spec.get("canonical_state_topic") or f"robot/{name}/state"),
            qos=str(spec.get("qos") or "reliable"),
            master_command=None,
            msg_type=str(spec.get("msg_type")) if spec.get("msg_type") else None,
            decoder=str(spec.get("decoder")) if spec.get("decoder") else None,
            eef_position_field=str(spec.get("eef_position_field")) if spec.get("eef_position_field") else None,
            eef_orientation_field=str(spec.get("eef_orientation_field")) if spec.get("eef_orientation_field") else None,
            gripper_field=str(spec.get("gripper_field")) if spec.get("gripper_field") else None,
        )
        for name, spec in profile.robot_states.items()
        if isinstance(spec, dict) and spec.get("topic")
    }
    return Ros2AdapterConfig(
        type="ros2_profile",
        ros_domain_id=int(connection.get("ros_domain_id") or 0),
        spin_threads=int(connection.get("spin_threads") or 2),
        cameras=cameras,
        arms=arms,
        master_overlay_topic=None,
        enable_depth=False,
    )


def profile_to_vsa_topics(profile: RobotProfile) -> tuple[str | None, str | None]:
    bindings = profile.payload.get("ROBOLINEAGE_bindings")
    vsa = bindings.get("vsa") if isinstance(bindings, dict) and isinstance(bindings.get("vsa"), dict) else {}
    camera = vsa.get("camera_ros_topic") or vsa.get("ros_camera_topic")
    arm = vsa.get("arm_ros_topic") or vsa.get("ros_arm_topic")
    color = profile.color_stream()
    state = profile.robot_state_stream()
    if not camera and isinstance(color, dict):
        camera = color.get("topic")
    if not arm and isinstance(state, dict):
        arm = state.get("topic")
    return (str(camera) if camera else None, str(arm) if arm else None)


def _uses_arx_decoder(spec: Any) -> bool:
    return isinstance(spec, dict) and str(spec.get("decoder") or "").startswith("arx")


def _capability_flag(payload: dict[str, Any], key: str, *, default: bool) -> bool:
    capabilities = payload.get("capabilities")
    if isinstance(capabilities, dict) and key in capabilities:
        return bool(capabilities[key])
    return default


def _gripper_source_label(state: dict[str, Any]) -> str | None:
    gripper = state.get("gripper")
    if isinstance(gripper, dict):
        source_type = str(gripper.get("source_type") or gripper.get("type") or "").strip()
        field = str(gripper.get("field") or "").strip()
        topic = str(gripper.get("topic") or "").strip()
        decoder = str(gripper.get("decoder") or "").strip()
        if source_type and field:
            return f"{source_type}:{field}"
        if topic:
            return f"topic:{topic}"
        if decoder:
            return f"decoder:{decoder}"
    if state.get("gripper_field"):
        return str(state["gripper_field"])
    if state.get("decoder"):
        return f"decoder:{state['decoder']}"
    return None
