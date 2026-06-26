"""Orchestrator — wires Config → adapters + SyncManager + Recorder.

`adapter_factory(name, spec_dict)` is an injected callable that returns a
DeviceAdapter for a given entry in the config. Production callers pass the
default factory (which knows how to build RealSenseAdapter, MockAdapter, etc.);
tests inject a mock factory.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Literal, Optional

_LOG = logging.getLogger(__name__)

from robolineage_data_source.adapters.base import DeviceAdapter
from robolineage_data_source.adapters.mock import MockAdapter
from robolineage_data_source.config.schema import Config
from robolineage_data_source.sync.manager import SyncManager
from robolineage_data_source.sync.registry import DeviceRegistry


AdapterFactory = Callable[[str, dict], DeviceAdapter]
RecorderMode = Literal["none", "rosbag"]


def default_adapter_factory(name: str, spec: dict) -> DeviceAdapter:
    """Build an adapter from a config entry.

    Supported `type` values:
        "realsense" → RealSenseAdapter   (requires pyrealsense2)
        "mock"      → MockAdapter
    Unknown types raise ValueError.
    """
    kind = spec.get("type")
    if kind == "realsense":
        from robolineage_data_source.adapters.cameras.realsense import RealSenseAdapter
        return RealSenseAdapter(
            name=name,
            serial=spec["serial"],
            resolution=tuple(spec.get("resolution", (1280, 720))),
            fps=int(spec.get("fps", 30)),
            depth=bool(spec.get("depth", False)),
        )
    if kind == "mock":
        return MockAdapter(
            topic=f"cam/{name}/color",
            rate_hz=float(spec.get("rate_hz", 30.0)),
        )
    raise ValueError(f"unknown adapter type: {kind!r}")


def create_rosbag_raw_recorder(
    *,
    rollout_dir: str | Path,
    topics: tuple[str, ...],
    ros_domain_id: int | None = None,
    storage_id: str | None = "mcap",
) -> Any:
    """Create the per-rollout direct ROS2 bag recorder."""
    from robolineage_data_source.rosbag import RosbagRawRecorder

    return RosbagRawRecorder(
        rollout_dir=rollout_dir,
        topics=topics,
        ros_domain_id=ros_domain_id,
        storage_id=storage_id,
    )


class Orchestrator:
    def __init__(
        self,
        config: Config,
        adapter_factory: AdapterFactory = default_adapter_factory,
        rollout_id: Optional[str] = None,
        recorder_mode: RecorderMode = "none",
    ) -> None:
        """Build the orchestrator.

        Args:
            config: parsed Config
            adapter_factory: per-device adapter constructor (legacy
                cameras/robots/imu mode); ignored when ``config.adapter`` is
                set (single-adapter ROS2 mode).
            rollout_id: explicit rollout id; auto-generated when None.
            recorder_mode: "none" | "rosbag".
        """
        if recorder_mode not in ("none", "rosbag"):
            raise ValueError(f"unknown recorder_mode: {recorder_mode!r}")
        self._config = config
        self._factory = adapter_factory
        self.rollout_id = rollout_id or uuid.uuid4().hex[:12]
        self._recorder_mode = recorder_mode
        self._registry = DeviceRegistry()

        self._adapters: dict[str, DeviceAdapter] = {}
        if config.adapter is not None:
            if config.adapter.type in {"ros2_arx_one", "ros2_profile"}:
                from robolineage_data_source.adapters.ros2_profile import Ros2ProfileAdapter

                self._adapters[config.adapter.type] = Ros2ProfileAdapter(
                    config=config.adapter,
                )
            else:
                raise ValueError(f"unknown adapter.type: {config.adapter.type!r}")
        else:
            for name, cam in config.cameras.items():
                spec = {
                    "type": cam.type,
                    "serial": cam.serial,
                    "resolution": cam.resolution,
                    "fps": cam.fps,
                    "depth": cam.depth,
                    **cam.extra,
                }
                self._adapters[name] = self._factory(name, spec)
            for name, robot in config.robots.items():
                spec = {"type": robot.type, "poll_rate": robot.poll_rate, **robot.extra}
                self._adapters[name] = self._factory(name, spec)
            for name, imu in config.imu.items():
                spec = {"type": imu.type, "port": imu.port, "rate": imu.rate, **imu.extra}
                self._adapters[name] = self._factory(name, spec)

        self._sync_manager = SyncManager(
            registry=self._registry,
            adapters=self._adapters,
            groups=config.sync_groups,
        )

        self._recorder = None
        if config.recorder is not None and recorder_mode == "rosbag":
            out_dir = Path(config.recorder.output_dir) / self.rollout_id
            self._recorder = create_rosbag_raw_recorder(
                rollout_dir=out_dir,
                topics=_rosbag_record_topics(config),
                ros_domain_id=config.adapter.ros_domain_id if config.adapter is not None else None,
            )

        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Orchestrator already started")
        try:
            if self._recorder is not None:
                self._recorder.start()
            self._sync_manager.start()
        except Exception:
            if self._recorder is not None:
                try:
                    self._recorder.stop()
                except Exception:
                    _LOG.exception("rollback recorder stop failed")
            raise
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        # Wrap each shutdown step so a single failure does not skip the others.
        try:
            self._sync_manager.stop()
        except Exception:
            _LOG.exception("sync manager stop failed; continuing recorder shutdown")
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:
                _LOG.exception("recorder stop failed")
        self._started = False

    def latest_camera_frame(
        self,
        *,
        stream_id: str | None = None,
        topic: str | None = None,
    ) -> Any | None:
        for adapter in self._adapters.values():
            getter = getattr(adapter, "latest_camera_frame", None)
            if not callable(getter):
                continue
            frame = getter(stream_id=stream_id, topic=topic)
            if frame is not None:
                return frame
        return None

    def camera_status(
        self,
        *,
        stream_id: str | None = None,
        topic: str | None = None,
    ) -> dict[str, Any] | None:
        for adapter in self._adapters.values():
            getter = getattr(adapter, "camera_status", None)
            if not callable(getter):
                continue
            status = getter(stream_id=stream_id, topic=topic)
            if status is not None:
                return status
        return None

    def latest_arm_vector(
        self,
        *,
        stream_id: str | None = None,
        topic: str | None = None,
    ) -> Any | None:
        for adapter in self._adapters.values():
            getter = getattr(adapter, "latest_arm_vector", None)
            if not callable(getter):
                continue
            vec = getter(stream_id=stream_id, topic=topic)
            if vec is not None:
                return vec
        return None

    def arm_status(
        self,
        *,
        stream_id: str | None = None,
        topic: str | None = None,
    ) -> dict[str, Any] | None:
        for adapter in self._adapters.values():
            getter = getattr(adapter, "arm_status", None)
            if not callable(getter):
                continue
            status = getter(stream_id=stream_id, topic=topic)
            if status is not None:
                return status
        return None

    def _video_profile(self) -> tuple[int, int, float]:
        if self._config.cameras:
            cam = self._config.cameras.get(
                "camera_h",
                next(iter(self._config.cameras.values())),
            )
            return cam.resolution[0], cam.resolution[1], float(cam.fps)
        return 1280, 720, 30.0


def _recorder_camera_names(config: Config) -> tuple[str, ...] | None:
    if config.recorder is not None and config.recorder.camera_names:
        return tuple(config.recorder.camera_names)
    if config.adapter is not None and config.adapter.cameras:
        names: list[str] = []
        for key, spec in config.adapter.cameras.items():
            name = spec.camera_name or key
            if name not in names:
                names.append(name)
        return _canonical_camera_names_or_none(names)
    if config.cameras:
        return _canonical_camera_names_or_none(config.cameras.keys())
    return None


def _canonical_camera_names_or_none(names: Any) -> tuple[str, ...] | None:
    aliases = {
        "head": "camera_h",
        "left_wrist": "camera_l",
        "right_wrist": "camera_r",
        "camera_h": "camera_h",
        "camera_l": "camera_l",
        "camera_r": "camera_r",
    }
    out: list[str] = []
    for raw in names:
        value = aliases.get(str(raw).strip())
        if value is None:
            return None
        if value not in out:
            out.append(value)
    return tuple(out) if out else None


def _rosbag_record_topics(config: Config) -> tuple[str, ...]:
    topics: list[str] = []
    if config.adapter is not None:
        selected_cameras = _recorder_camera_name_filter(config)
        for name, spec in config.adapter.cameras.items():
            aliases = {
                str(name),
                str(spec.camera_name or ""),
                str(spec.stream_id or ""),
            }
            if selected_cameras is None or aliases & selected_cameras:
                topics.append(spec.topic)
        for spec in config.adapter.arms.values():
            topics.append(spec.slave_status)
            if spec.master_command:
                topics.append(spec.master_command)
    unique = tuple(dict.fromkeys(str(topic).strip() for topic in topics if str(topic).strip()))
    if not unique:
        raise ValueError("direct rosbag recording requires ROS2 adapter topics")
    return unique


def _recorder_camera_name_filter(config: Config) -> set[str] | None:
    if config.recorder is None or not config.recorder.camera_names:
        return None
    raw = {str(item).strip() for item in config.recorder.camera_names if str(item).strip()}
    if not raw:
        return None
    canonical = _canonical_camera_names_or_none(raw)
    return raw | set(canonical or ())
