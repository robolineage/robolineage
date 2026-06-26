"""Pure-Python helpers for RealSense adapter — isolated so they can be tested
without the SDK.

The frame-adapter functions accept any object that looks like a
`pyrealsense2.video_frame` (duck-typed). This keeps the helpers testable with a
stub and lets us unit-test Sample construction end-to-end without hardware.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from robolineage_data_source.sample import Sample

# Inter-camera sync mode values as defined by librealsense
# (rs2_option::RS2_OPTION_INTER_CAM_SYNC_MODE); matches rs.option.inter_cam_sync_mode.
RS_SYNC_MODE_DEFAULT = 0  # no sync
RS_SYNC_MODE_MASTER = 1
RS_SYNC_MODE_SLAVE = 2

_ROLE_TO_MODE = {
    "master": RS_SYNC_MODE_MASTER,
    "slave": RS_SYNC_MODE_SLAVE,
    "none": RS_SYNC_MODE_DEFAULT,
}


def sync_mode_for_role(role: str) -> int:
    """Translate a high-level role ("master"/"slave"/"none") into the
    librealsense INTER_CAM_SYNC_MODE integer. Raises ValueError on unknown role.
    """
    try:
        return _ROLE_TO_MODE[role]
    except KeyError:
        raise ValueError(
            f"unknown sync role {role!r}; expected one of {sorted(_ROLE_TO_MODE)}"
        )


def _read_metadata(frame: Any, key: str, default=None):
    """Read a frame-metadata field by symbolic name, returning `default` if unsupported.

    We use string keys because `pyrealsense2.frame_metadata_value` is only
    available in SDK-equipped environments. The RealSenseAdapter passes the
    real SDK enum when available; tests pass strings. `supports_frame_metadata`
    returns False when the SDK can't provide the field (common on older firmware).

    The `except (KeyError, RuntimeError)` clause handles two distinct cases:
      - KeyError: the test stub raises this for unknown keys; the SDK never
        raises KeyError.
      - RuntimeError: the pyrealsense2 SDK's own exception type. Raised in rare
        cases where `supports_frame_metadata` returns True but `get_frame_metadata`
        then fails (firmware glitch, concurrent sensor disconnect). We deliberately
        collapse both into `default` — stream-level faults are observable via
        the adapter's health() status; per-sample metadata failures shouldn't
        kill the frame loop.
    """
    try:
        if hasattr(frame, "supports_frame_metadata") and not frame.supports_frame_metadata(key):
            return default
        return frame.get_frame_metadata(key)
    except (KeyError, RuntimeError):
        return default


def _hw_timestamp_ns(frame: Any) -> int | None:
    """Return the RealSense sensor timestamp in nanoseconds, or None if unavailable.

    The SDK reports sensor_timestamp in microseconds as an integer; we multiply
    by 1000 to normalize to `device_hw_ns` nanoseconds (same convention as
    every other adapter).
    """
    raw_us = _read_metadata(frame, "sensor_timestamp", default=None)
    if raw_us is None:
        return None
    return int(raw_us) * 1000


def _frame_data_copy(frame: Any) -> np.ndarray:
    """Detach frame bytes from SDK-owned memory before publishing.

    librealsense frame buffers are recycled aggressively. Keeping a view into
    `frame.get_data()` past the current iteration can therefore corrupt
    downstream consumers once the next frame arrives.
    """
    return np.asanyarray(frame.get_data()).copy()


def select_sync_sensor(
    sensors: Iterable[Any],
    *,
    option_key: Any,
    name_getter: Callable[[Any], str],
) -> Any:
    """Pick the sensor used to configure inter-camera sync.

    Preference order:
      1. A sync-capable sensor whose name starts with "stereo"
      2. Any sync-capable sensor
      3. A stereo-named sensor (useful so callers can emit a specific error)
      4. The first sensor, if any
    """
    sensor_list = list(sensors)
    if not sensor_list:
        raise RuntimeError("RealSense device exposes no sensors")

    sync_capable = [sensor for sensor in sensor_list if sensor.supports(option_key)]
    for sensor in sync_capable:
        if name_getter(sensor).lower().startswith("stereo"):
            return sensor
    if sync_capable:
        return sync_capable[0]
    for sensor in sensor_list:
        if name_getter(sensor).lower().startswith("stereo"):
            return sensor
    return sensor_list[0]


def apply_sync_mode(
    sensor: Any,
    *,
    option_key: Any,
    role: str,
    sensor_label: str,
) -> None:
    """Apply inter-camera sync mode and fail loudly when required support is absent."""
    mode = sync_mode_for_role(role)
    if not sensor.supports(option_key):
        if role == "none":
            return
        raise RuntimeError(
            f"{sensor_label} does not support inter_cam_sync_mode for role {role!r}"
        )
    sensor.set_option(option_key, float(mode))


def color_frame_to_sample(frame: Any, topic: str, host_mono_ns: int) -> Sample:
    """Build a Sample for a color (RGB/BGR) frame.

    Caller provides `host_mono_ns` from `time.monotonic_ns()` captured
    immediately after `pipeline.wait_for_frames()` returned — this is the
    authoritative host-side timestamp.
    """
    data = _frame_data_copy(frame)
    hw_ns = _hw_timestamp_ns(frame)
    meta: dict[str, Any] = {
        "width": frame.get_width(),
        "height": frame.get_height(),
        "exposure_us": _read_metadata(frame, "actual_exposure", default=None),
        "gain": _read_metadata(frame, "gain_level", default=None),
    }
    return Sample(
        topic=topic,
        host_mono_ns=host_mono_ns,
        payload=data,
        device_hw_ns=hw_ns,
        device_hw_domain="realsense_global" if hw_ns is not None else None,
        meta=meta,
    )


def depth_frame_to_sample(frame: Any, topic: str, host_mono_ns: int) -> Sample:
    """Build a Sample for a depth frame (uint16, in mm as provided by the SDK)."""
    data = _frame_data_copy(frame)
    hw_ns = _hw_timestamp_ns(frame)
    meta: dict[str, Any] = {
        "width": frame.get_width(),
        "height": frame.get_height(),
    }
    return Sample(
        topic=topic,
        host_mono_ns=host_mono_ns,
        payload=data,
        device_hw_ns=hw_ns,
        device_hw_domain="realsense_global" if hw_ns is not None else None,
        meta=meta,
    )
