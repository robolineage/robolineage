"""RealSenseAdapter — supervises one Intel RealSense device
Intel RealSense device, with hardware-synchronized inter-camera timestamps.

This module imports `pyrealsense2` lazily so the rest of the package remains
importable on machines without the SDK. Instantiate the adapter only when you
intend to talk to real hardware.

Raw capture is handled by ROS2 bag recording. This adapter remains as a
source-side liveness/sync supervisor for deployments that still need direct
RealSense startup checks.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from robolineage_data_source.adapters.base import DeviceAdapter
from robolineage_data_source.adapters.cameras.realsense_helpers import (
    apply_sync_mode,
    select_sync_sensor,
    sync_mode_for_role,
)
from robolineage_data_source.sample import HealthState, HealthStatus

_LOG = logging.getLogger(__name__)


class RealSenseAdapter(DeviceAdapter):
    """One Intel RealSense device. Threaded frame reader for health only.

    Args:
        name: Logical camera name used in topic prefix ("cam/{name}/...").
        serial: librealsense device serial (from `rs-enumerate-devices`).
        resolution: (width, height) tuple. Must match a supported profile.
        fps: Stream frame rate.
        depth: If True, also stream aligned depth.

    Lifecycle:
        configure_sync("master"|"slave"|"none")   # optional, BEFORE start
        start()
        stop()

    configure_sync must be called before start() because librealsense applies
    INTER_CAM_SYNC_MODE only at pipeline-start time.
    """

    def __init__(
        self,
        name: str,
        serial: str,
        resolution: tuple[int, int] = (1280, 720),
        fps: int = 30,
        depth: bool = False,
    ) -> None:
        self._name = name
        self._serial = serial
        self._resolution = resolution
        self._fps = fps
        self._depth = depth
        self._sync_role: str = "none"
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pipeline = None  # pyrealsense2.pipeline, populated in start()
        self._started = False
        self._last_mono: Optional[int] = None
        self._health_state: HealthState = HealthState.NOT_STARTED
        self._health_message: str = ""
        self._lock = threading.Lock()  # protects _health_* and _last_mono
        self._lifecycle_lock = threading.Lock()  # serializes start()/stop()

    def supports_hw_sync(self) -> bool:
        return True

    def configure_sync(self, role: str) -> None:
        # Validate before storing; raises ValueError on bad role.
        sync_mode_for_role(role)
        if self._started:
            raise RuntimeError(
                "configure_sync must be called before start(); "
                "librealsense applies INTER_CAM_SYNC_MODE at pipeline-start time"
            )
        self._sync_role = role

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError(f"RealSenseAdapter[{self._name}] already started")
            # Lazy import so the package stays importable without the SDK.
            import pyrealsense2 as rs

            cfg = rs.config()
            cfg.enable_device(self._serial)
            w, h = self._resolution
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, self._fps)
            if self._depth:
                cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, self._fps)

            self._pipeline = rs.pipeline()

            # Apply INTER_CAM_SYNC_MODE on the depth sensor BEFORE pipeline.start().
            # librealsense requires the option to be set on the first depth sensor
            # of the device; it takes effect when the pipeline is started.
            ctx = rs.context()
            device = None
            for d in ctx.query_devices():
                if d.get_info(rs.camera_info.serial_number) == self._serial:
                    device = d
                    break
            if device is None:
                raise RuntimeError(
                    f"RealSense device with serial {self._serial!r} not found"
                )
            sensors = list(device.query_sensors())
            depth_sensor = select_sync_sensor(
                sensors,
                option_key=rs.option.inter_cam_sync_mode,
                name_getter=lambda sensor: sensor.get_info(rs.camera_info.name),
            )
            apply_sync_mode(
                depth_sensor,
                option_key=rs.option.inter_cam_sync_mode,
                role=self._sync_role,
                sensor_label=(
                    f"RealSense device {self._serial!r} "
                    f"sensor {depth_sensor.get_info(rs.camera_info.name)!r}"
                ),
            )

            self._pipeline.start(cfg)
            self._stop_event.clear()
            self._started = True
            with self._lock:
                self._health_state = HealthState.DEGRADED  # until first frame
                self._health_message = "waiting for first frame"
                self._last_mono = None
            self._thread = threading.Thread(
                target=self._run,
                name=f"RealSenseAdapter[{self._name}]",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            if not self._started:
                return
            # Flip _started up front so a concurrent stop() that grabs the
            # lifecycle_lock after this one releases is a clean no-op.
            self._started = False
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=3.0)
                self._thread = None
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:
                    # Pipeline may already be torn down if frame loop errored; don't mask shutdown.
                    _LOG.exception("RealSenseAdapter[%s] pipeline.stop failed", self._name)
                self._pipeline = None
            with self._lock:
                self._health_state = HealthState.NOT_STARTED
                self._health_message = ""

    def health(self) -> HealthStatus:
        with self._lock:
            return HealthStatus(
                state=self._health_state,
                message=self._health_message,
                last_sample_mono_ns=self._last_mono,
            )

    def _run(self) -> None:
        """Frame loop. Runs until stop_event is set. On SDK errors, marks
        health FAILED and exits — the user is expected to notice via health().
        """
        import pyrealsense2 as rs

        assert self._pipeline is not None
        try:
            while not self._stop_event.is_set():
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError as exc:
                    # Timeout or transient SDK error — report degraded, continue.
                    with self._lock:
                        self._health_state = HealthState.DEGRADED
                        self._health_message = f"wait_for_frames: {exc}"
                    continue

                host_mono_ns = time.monotonic_ns()

                try:
                    color = frames.get_color_frame()
                    if color:
                        with self._lock:
                            self._last_mono = host_mono_ns
                            self._health_state = HealthState.OK
                            self._health_message = ""

                    if self._depth:
                        depth = frames.get_depth_frame()
                        if depth:
                            _ = depth
                except Exception as exc:
                    # Per-frame SDK failure — stay alive, mark degraded.
                    with self._lock:
                        self._health_state = HealthState.DEGRADED
                        self._health_message = f"publish/convert failed: {exc!r}"
        except Exception as exc:
            with self._lock:
                self._health_state = HealthState.FAILED
                self._health_message = f"frame loop crashed: {exc!r}"


class _FrameMetaShim:
    """Adapts a pyrealsense2.video_frame's `get_frame_metadata(rs.frame_metadata_value.X)`
    API into the string-keyed interface the helper functions use.

    This shim exists only so `realsense_helpers` stays SDK-free. It maps
    `"sensor_timestamp"` → `rs.frame_metadata_value.sensor_timestamp` etc.
    """
    # If helpers start calling new metadata-keyed methods, add the enum mapping here.
    _KEYS = {
        "sensor_timestamp": "sensor_timestamp",
        "actual_exposure": "actual_exposure",
        "gain_level": "gain_level",
    }

    def __init__(self, frame, rs_module):
        self._frame = frame
        self._rs = rs_module

    def __getattr__(self, name):
        return getattr(self._frame, name)

    def get_data(self):
        return self._frame.get_data()

    def get_width(self):
        return self._frame.get_width()

    def get_height(self):
        return self._frame.get_height()

    def _enum(self, key):
        return getattr(self._rs.frame_metadata_value, self._KEYS[key])

    def supports_frame_metadata(self, key):
        return self._frame.supports_frame_metadata(self._enum(key))

    def get_frame_metadata(self, key):
        return self._frame.get_frame_metadata(self._enum(key))
