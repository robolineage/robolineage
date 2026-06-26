"""Typed configuration schema for the data-source layer.

A single YAML file drives the entire system. Each top-level section maps to
one of these dataclasses. See `configs/ROBOLINEAGE_default.yaml`,
`configs/robot_profiles/`, and `docs/deployment/` for canonical examples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutConfig:
    task_id: str
    operator_id: str
    mode: str = "C1"
    policy_version: str | None = None


@dataclass
class SyncGroupConfig:
    name: str
    backend: str                 # e.g. "realsense_inter_cam"
    master: str                  # name of the master device
    slaves: list[str] = field(default_factory=list)


@dataclass
class CameraConfig:
    type: str                    # "realsense" | "zed" | "v4l2" | "gopro" | "rtsp"
    serial: str | None = None
    resolution: tuple[int, int] = (1280, 720)
    fps: int = 30
    depth: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class IMUConfig:
    type: str                    # "serial" | ...
    port: str | None = None
    rate: int = 200
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotConfig:
    type: str                    # robot-specific adapter type for legacy per-device mode
    poll_rate: int = 200
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecorderConfig:
    output_dir: str
    camera_names: tuple[str, ...] | None = None


@dataclass
class PreviewConfig:
    bind: str = "0.0.0.0:8080"
    stream_bitrate: int = 5_000_000


@dataclass(frozen=True)
class CameraTopicSpec:
    """ROS2 camera subscription mapped to one logical stream id."""

    topic: str
    transport: str
    qos: str
    stream_id: str
    camera_name: str | None = None


@dataclass(frozen=True)
class ArmTopicSpec:
    """ROS2 robot-state subscription mapped to a logical state stream.

    The realtime VSA consumes a compact 27-value vector decoded from ROS2.
    ``msg_type`` / decoder fields let robot profiles describe how a concrete
    ROS2 message should be converted to that vector without hard-coding a robot
    family into the main config.
    """

    slave_status: str
    state_stream_id: str
    qos: str = "reliable"
    master_command: str | None = None
    msg_type: str | None = None
    decoder: str | None = None
    eef_position_field: str | None = None
    eef_orientation_field: str | None = None
    gripper_field: str | None = None


@dataclass
class Ros2AdapterConfig:
    """Top-level ROS2 adapter config.

    One profile-driven ROS2 adapter instance serves all configured cameras and
    robot-state streams.
    """

    type: str = "ros2_profile"
    ros_domain_id: int = 0
    spin_threads: int = 2
    cameras: dict[str, CameraTopicSpec] = field(default_factory=dict)
    arms: dict[str, ArmTopicSpec] = field(default_factory=dict)
    master_overlay_topic: str | None = None
    enable_depth: bool = False


@dataclass
class ServicesToggle:
    """v0.2.0+ Phase 5 — operator-tunable on/off for unified launcher subsystems.

    All default on. Used by `robolineage_app.UnifiedRuntime` to decide which sub-runner
    to start. Setting `vsa: false` is the typical "data-source-only smoke test"
    config; `health_check: false` is for setups that proxy /health elsewhere.
    """
    data_source: bool = True
    session: bool = True
    vsa: bool = True
    post_review: bool = True
    health_check: bool = True


@dataclass
class TuningConfig:
    """v0.2.0+ Phase 5 — operator-tunable runtime knobs.

    Defaults keep online VSA conservative while reducing per-call image cost.
    """
    ring_capacity: int = 120
    still_threshold: float = 3e-4
    still_min_frames: int = 25        # raised from 15 (~0.5s) to 25 (~0.83s@30Hz)
    gripper_close_threshold: float = -1.0
    rotation_weight: float = 0.2
    smoothing_window: int = 10
    motion_resume_threshold: float = 8e-4
    context_frames: int = 15
    max_keyframes: int = 3
    idle_timeout: float = 10.0
    periodic_interval_sec: float = 2.0
    heartbeat_interval: float = 5.0
    merge_window_sec: float = 1.0
    final_settle_sec: float = 1.0
    max_vlm_windows_per_rollout: int | None = None
    min_same_event_interval: float = 3.0
    vlm_workers: int = 1
    strong_prior_margin: float = 0.35   # PhaseFusion rule-4 threshold; raised from 0.25
    prior_sticky_frames: int = 2        # consecutive frames required before prior overrides VLM


@dataclass
class VsaConfig:
    """v0.2.0+ Phase 5 — VSA realtime entrypoint configuration."""
    camera_topic: str = "camera/primary/color"
    arm_topic: str = "robot/active/state"
    task_config_path: str | None = None
    output_jsonl_path: str | None = None
    max_events: int | None = None


@dataclass
class VlmConfig:
    """v0.2.0+ Phase 5 — VSA VLM tunables.

    Model / key / endpoint come from environment variables (VSA_VLM_*),
    falling back to OPENAI_* for backward compatibility.  Only latency
    tunables are kept here so the yaml stays secret-free.

    Env vars consumed by runtime.py:
        VSA_VLM_MODEL    — model name (e.g. google/gemini-2.0-flash)
        VSA_VLM_API_KEY  — API key
        VSA_VLM_BASE_URL — base URL for proxy / gateway
        VSA_VLM_TIMEOUT  — per-call timeout in seconds (overrides yaml)
        VSA_VLM_MAX_TOKENS — max output tokens (overrides yaml)
    Fallbacks: OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL
    """
    timeout: float = 20.0
    max_output_tokens: int = 256


@dataclass
class PostReviewConfig:
    """Post-rollout review worker configuration.

    The worker runs off the realtime path. VLM calls use the same backend
    configuration as online VSA by default, but are gated so they pause while
    an online rollout is active.
    """
    use_vlm: bool = True
    idle_delay_sec: float = 5.0
    max_review_images: int = 12


@dataclass
class HealthConfig:
    """v0.2.0+ Phase 5 — /health endpoint configuration."""
    port: int = 8081
    bind: str = "0.0.0.0"


@dataclass
class Config:
    rollout: RolloutConfig
    robot_profile_path: str | None = None
    sync_groups: list[SyncGroupConfig] = field(default_factory=list)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    imu: dict[str, IMUConfig] = field(default_factory=dict)
    robots: dict[str, RobotConfig] = field(default_factory=dict)
    recorder: RecorderConfig | None = None
    preview: PreviewConfig | None = None
    adapter: Ros2AdapterConfig | None = None
    services: ServicesToggle | None = None
    tuning: TuningConfig | None = None
    vsa: VsaConfig | None = None
    vlm: VlmConfig | None = None
    post_review: PostReviewConfig | None = None
    health: HealthConfig | None = None
