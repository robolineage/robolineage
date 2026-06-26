"""YAML → Config loader. Uses `yaml.safe_load` — never execute arbitrary YAML tags."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robolineage_data_source.config.schema import (
    ArmTopicSpec,
    CameraConfig,
    CameraTopicSpec,
    Config,
    HealthConfig,
    IMUConfig,
    PreviewConfig,
    PostReviewConfig,
    RecorderConfig,
    RobotConfig,
    Ros2AdapterConfig,
    RolloutConfig,
    ServicesToggle,
    SyncGroupConfig,
    TuningConfig,
    VlmConfig,
    VsaConfig,
)

_KNOWN_CAMERA_FIELDS = {"type", "serial", "resolution", "fps", "depth"}
_KNOWN_IMU_FIELDS = {"type", "port", "rate"}
_KNOWN_ROBOT_FIELDS = {"type", "poll_rate"}


def load_config(path: str | Path) -> Config:
    """Parse a YAML file into a `Config`. Raises ValueError on structural errors."""
    config_path = Path(path).expanduser()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config must be a YAML mapping at the top level; got "
            f"{type(raw).__name__}"
        )

    if "rollout" not in raw:
        raise ValueError("Config missing required section: 'rollout'")

    adapter = None
    if "adapter" in raw:
        adapter = _parse_adapter(
            raw["adapter"],
            raw.get("cameras_ros2", {}) or {},
            raw.get("arms_ros2", {}) or {},
        )

    return Config(
        rollout=_parse_rollout(raw["rollout"]),
        robot_profile_path=_resolve_optional_path(
            raw.get("robot_profile_path"),
            base_dir=config_path.parent,
        ),
        sync_groups=[_parse_sync_group(g) for g in raw.get("sync_groups", []) or []],
        cameras={n: _parse_camera(c) for n, c in (raw.get("cameras") or {}).items()},
        imu={n: _parse_imu(i) for n, i in (raw.get("imu") or {}).items()},
        robots={n: _parse_robot(r) for n, r in (raw.get("robots") or {}).items()},
        recorder=_parse_recorder(raw["recorder"]) if "recorder" in raw else None,
        preview=_parse_preview(raw["preview"]) if "preview" in raw else None,
        adapter=adapter,
        services=_parse_services(raw["services"]) if "services" in raw else None,
        tuning=_parse_tuning(raw["tuning"]) if "tuning" in raw else None,
        vsa=_parse_vsa(raw["vsa"]) if "vsa" in raw else None,
        vlm=_parse_vlm(raw["vlm"]) if "vlm" in raw else None,
        post_review=(
            _parse_post_review(raw["post_review"])
            if "post_review" in raw
            else None
        ),
        health=_parse_health(raw["health"]) if "health" in raw else None,
    )


def _parse_rollout(d: dict[str, Any]) -> RolloutConfig:
    return RolloutConfig(
        task_id=d["task_id"],
        operator_id=d["operator_id"],
        mode=d.get("mode", "C1"),
        policy_version=d.get("policy_version"),
    )


def _parse_sync_group(d: dict[str, Any]) -> SyncGroupConfig:
    return SyncGroupConfig(
        name=d["name"],
        backend=d["backend"],
        master=d["master"],
        slaves=list(d.get("slaves", []) or []),
    )


def _parse_camera(d: dict[str, Any]) -> CameraConfig:
    extra = {k: v for k, v in d.items() if k not in _KNOWN_CAMERA_FIELDS}
    res = d.get("resolution", [1280, 720])
    return CameraConfig(
        type=d["type"],
        serial=d.get("serial"),
        resolution=(int(res[0]), int(res[1])),
        fps=int(d.get("fps", 30)),
        depth=bool(d.get("depth", False)),
        extra=extra,
    )


def _parse_imu(d: dict[str, Any]) -> IMUConfig:
    extra = {k: v for k, v in d.items() if k not in _KNOWN_IMU_FIELDS}
    return IMUConfig(
        type=d["type"],
        port=d.get("port"),
        rate=int(d.get("rate", 200)),
        extra=extra,
    )


def _parse_robot(d: dict[str, Any]) -> RobotConfig:
    extra = {k: v for k, v in d.items() if k not in _KNOWN_ROBOT_FIELDS}
    return RobotConfig(
        type=d["type"],
        poll_rate=int(d.get("poll_rate", 200)),
        extra=extra,
    )


def _parse_recorder(d: dict[str, Any]) -> RecorderConfig:
    camera_names_raw = d.get("camera_names")
    camera_names = None
    if camera_names_raw is not None:
        if not isinstance(camera_names_raw, list):
            raise ValueError("recorder.camera_names must be a list when provided")
        camera_names = tuple(str(item) for item in camera_names_raw if str(item).strip())
    return RecorderConfig(
        output_dir=d["output_dir"],
        camera_names=camera_names,
    )


def _parse_preview(d: dict[str, Any]) -> PreviewConfig:
    return PreviewConfig(
        bind=d.get("bind", "0.0.0.0:8080"),
        stream_bitrate=int(d.get("stream_bitrate", 5_000_000)),
    )


def _parse_camera_topic(d: dict[str, Any]) -> CameraTopicSpec:
    return CameraTopicSpec(
        topic=str(d["topic"]),
        transport=str(d.get("transport", "compressed")),
        qos=str(d.get("qos", "sensor_data")),
        stream_id=str(d["stream_id"]),
        camera_name=str(d["camera_name"]) if d.get("camera_name") else None,
    )


def _parse_arm_topic(d: dict[str, Any]) -> ArmTopicSpec:
    return ArmTopicSpec(
        slave_status=str(d["slave_status"]),
        state_stream_id=str(d["state_stream_id"]),
        qos=str(d.get("qos", "reliable")),
        master_command=d.get("master_command"),
        msg_type=d.get("msg_type"),
        decoder=d.get("decoder"),
        eef_position_field=d.get("eef_position_field"),
        eef_orientation_field=d.get("eef_orientation_field"),
        gripper_field=d.get("gripper_field"),
    )


def _parse_adapter(
    adapter_d: dict[str, Any],
    cams_d: dict[str, Any],
    arms_d: dict[str, Any],
) -> Ros2AdapterConfig:
    return Ros2AdapterConfig(
        type=str(adapter_d.get("type", "ros2_profile")),
        ros_domain_id=int(adapter_d.get("ros_domain_id", 0)),
        spin_threads=int(adapter_d.get("spin_threads", 2)),
        cameras={n: _parse_camera_topic(c) for n, c in cams_d.items()},
        arms={n: _parse_arm_topic(a) for n, a in arms_d.items()},
        master_overlay_topic=adapter_d.get("master_overlay_topic"),
        enable_depth=bool(adapter_d.get("enable_depth", False)),
    )


def _parse_services(d: dict[str, Any]) -> ServicesToggle:
    return ServicesToggle(
        data_source=bool(d.get("data_source", True)),
        session=bool(d.get("session", True)),
        vsa=bool(d.get("vsa", True)),
        post_review=bool(d.get("post_review", True)),
        health_check=bool(d.get("health_check", True)),
    )


def _parse_tuning(d: dict[str, Any]) -> TuningConfig:
    return TuningConfig(
        ring_capacity=int(d.get("ring_capacity", 120)),
        still_threshold=float(d.get("still_threshold", 3e-4)),
        still_min_frames=int(d.get("still_min_frames", 25)),
        gripper_close_threshold=float(d.get("gripper_close_threshold", -1.0)),
        rotation_weight=float(d.get("rotation_weight", 0.2)),
        smoothing_window=int(d.get("smoothing_window", 10)),
        motion_resume_threshold=float(d.get("motion_resume_threshold", 8e-4)),
        context_frames=int(d.get("context_frames", 15)),
        max_keyframes=int(d.get("max_keyframes", 3)),
        idle_timeout=float(d.get("idle_timeout", 10.0)),
        periodic_interval_sec=float(d.get("periodic_interval_sec", 2.0)),
        heartbeat_interval=float(d.get("heartbeat_interval", 5.0)),
        merge_window_sec=float(d.get("merge_window_sec", 1.0)),
        final_settle_sec=float(d.get("final_settle_sec", 1.0)),
        max_vlm_windows_per_rollout=(
            int(d["max_vlm_windows_per_rollout"])
            if d.get("max_vlm_windows_per_rollout") is not None
            else None
        ),
        min_same_event_interval=float(d.get("min_same_event_interval", 3.0)),
        vlm_workers=int(d.get("vlm_workers", 1)),
        strong_prior_margin=float(d.get("strong_prior_margin", 0.35)),
        prior_sticky_frames=int(d.get("prior_sticky_frames", 2)),
    )


def _parse_vsa(d: dict[str, Any]) -> VsaConfig:
    return VsaConfig(
        camera_topic=str(d.get("camera_topic", "camera/primary/color")),
        arm_topic=str(d.get("arm_topic", "robot/active/state")),
        task_config_path=d.get("task_config_path"),
        output_jsonl_path=d.get("output_jsonl_path"),
        max_events=d.get("max_events"),
    )


def _parse_vlm(d: dict[str, Any]) -> VlmConfig:
    return VlmConfig(
        timeout=float(d.get("timeout", 20.0)),
        max_output_tokens=int(d.get("max_output_tokens", 256)),
    )


def _parse_post_review(d: dict[str, Any]) -> PostReviewConfig:
    return PostReviewConfig(
        use_vlm=bool(d.get("use_vlm", True)),
        idle_delay_sec=float(d.get("idle_delay_sec", 5.0)),
        max_review_images=int(d.get("max_review_images", 12)),
    )


def _parse_health(d: dict[str, Any]) -> HealthConfig:
    return HealthConfig(
        port=int(d.get("port", 8081)),
        bind=str(d.get("bind", "0.0.0.0")),
    )


def _resolve_optional_path(value: Any, *, base_dir: Path) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    candidate = base_dir / path
    if candidate.exists():
        return str(candidate.resolve())
    return str((Path.cwd() / path).resolve())
