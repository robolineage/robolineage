"""Tests for RealSense pure helpers — no SDK needed."""
import numpy as np
import pytest

from robolineage_data_source.adapters.cameras.realsense_helpers import (
    RS_SYNC_MODE_DEFAULT,
    RS_SYNC_MODE_MASTER,
    RS_SYNC_MODE_SLAVE,
    apply_sync_mode,
    color_frame_to_sample,
    depth_frame_to_sample,
    select_sync_sensor,
    sync_mode_for_role,
)


def test_sync_mode_for_role_master():
    assert sync_mode_for_role("master") == RS_SYNC_MODE_MASTER


def test_sync_mode_for_role_slave():
    assert sync_mode_for_role("slave") == RS_SYNC_MODE_SLAVE


def test_sync_mode_for_role_none():
    assert sync_mode_for_role("none") == RS_SYNC_MODE_DEFAULT


def test_sync_mode_for_role_unknown_raises():
    with pytest.raises(ValueError, match="unknown sync role"):
        sync_mode_for_role("leader")


class _StubFrame:
    """Minimal stand-in for pyrealsense2.video_frame — just enough for helpers."""
    def __init__(
        self,
        data: np.ndarray,
        sensor_ts_us: float,
        host_mono_ns: int,
        exposure_us: int = 8000,
        gain: int = 16,
    ):
        self._data = data
        self._sensor_ts_us = sensor_ts_us
        self._host_mono_ns = host_mono_ns
        self._exposure = exposure_us
        self._gain = gain

    def get_data(self):
        return self._data

    def get_frame_metadata(self, key):
        if key == "sensor_timestamp":
            return int(self._sensor_ts_us)
        if key == "actual_exposure":
            return self._exposure
        if key == "gain_level":
            return self._gain
        raise KeyError(key)

    def supports_frame_metadata(self, key):
        return key in {"sensor_timestamp", "actual_exposure", "gain_level"}

    def get_width(self):
        return self._data.shape[1]

    def get_height(self):
        return self._data.shape[0]


def test_color_frame_to_sample():
    arr = np.zeros((720, 1280, 3), dtype=np.uint8)
    arr[0, 0] = [255, 128, 64]
    frame = _StubFrame(arr, sensor_ts_us=123_456_789, host_mono_ns=9_000_000_000)
    sample = color_frame_to_sample(frame, topic="cam/h/color", host_mono_ns=9_000_000_000)

    assert sample.topic == "cam/h/color"
    assert sample.host_mono_ns == 9_000_000_000
    assert sample.device_hw_ns == 123_456_789_000  # µs → ns
    assert sample.device_hw_domain == "realsense_global"
    assert sample.payload.shape == (720, 1280, 3)
    assert sample.payload.dtype == np.uint8
    np.testing.assert_array_equal(sample.payload[0, 0], [255, 128, 64])
    assert sample.meta["exposure_us"] == 8000
    assert sample.meta["gain"] == 16
    assert sample.meta["width"] == 1280
    assert sample.meta["height"] == 720
    arr[0, 0] = [1, 2, 3]
    np.testing.assert_array_equal(sample.payload[0, 0], [255, 128, 64])


def test_depth_frame_to_sample():
    arr = np.full((720, 1280), 1500, dtype=np.uint16)
    frame = _StubFrame(arr, sensor_ts_us=123_500_000, host_mono_ns=9_100_000_000)
    sample = depth_frame_to_sample(frame, topic="cam/h/depth", host_mono_ns=9_100_000_000)

    assert sample.topic == "cam/h/depth"
    assert sample.payload.dtype == np.uint16
    assert sample.payload.shape == (720, 1280)
    assert sample.device_hw_ns == 123_500_000_000
    assert sample.meta["width"] == 1280


def test_color_frame_missing_metadata_ok():
    """If sensor_timestamp metadata is unavailable, device_hw_ns is None."""
    arr = np.zeros((480, 640, 3), dtype=np.uint8)

    class _NoMetaFrame(_StubFrame):
        def supports_frame_metadata(self, key):
            return False

    frame = _NoMetaFrame(arr, sensor_ts_us=0, host_mono_ns=1)
    sample = color_frame_to_sample(frame, topic="cam/h/color", host_mono_ns=1)
    assert sample.device_hw_ns is None
    assert sample.device_hw_domain is None


def test_color_frame_metadata_read_raises_falls_back_to_none():
    """If supports_frame_metadata lies (returns True) but the read raises,
    helpers must fall back to None rather than propagate the SDK error."""
    arr = np.zeros((480, 640, 3), dtype=np.uint8)

    class _RaisingFrame(_StubFrame):
        def supports_frame_metadata(self, key):
            return True  # lie — claim support

        def get_frame_metadata(self, key):
            raise RuntimeError("simulated SDK read failure")

    frame = _RaisingFrame(arr, sensor_ts_us=0, host_mono_ns=1)
    sample = color_frame_to_sample(frame, topic="cam/h/color", host_mono_ns=1)
    assert sample.device_hw_ns is None
    assert sample.device_hw_domain is None
    # exposure/gain also fall through to None but shouldn't crash
    assert sample.meta["exposure_us"] is None
    assert sample.meta["gain"] is None


class _StubSensor:
    def __init__(self, name: str, supports_sync: bool):
        self.name = name
        self._supports_sync = supports_sync
        self.set_calls = []

    def supports(self, key):
        return self._supports_sync

    def set_option(self, key, value):
        self.set_calls.append((key, value))


def test_select_sync_sensor_prefers_stereo_sensor_that_supports_sync():
    option_key = object()
    sensors = [
        _StubSensor(name="RGB Camera", supports_sync=True),
        _StubSensor(name="Stereo Module", supports_sync=True),
    ]
    chosen = select_sync_sensor(
        sensors,
        option_key=option_key,
        name_getter=lambda sensor: sensor.name,
    )
    assert chosen is sensors[1]


def test_select_sync_sensor_falls_back_to_any_sync_capable_sensor():
    option_key = object()
    sensors = [
        _StubSensor(name="RGB Camera", supports_sync=False),
        _StubSensor(name="Motion Module", supports_sync=True),
    ]
    chosen = select_sync_sensor(
        sensors,
        option_key=option_key,
        name_getter=lambda sensor: sensor.name,
    )
    assert chosen is sensors[1]


def test_apply_sync_mode_sets_option_when_supported():
    option_key = object()
    sensor = _StubSensor(name="Stereo Module", supports_sync=True)
    apply_sync_mode(
        sensor,
        option_key=option_key,
        role="master",
        sensor_label="camera_h stereo",
    )
    assert sensor.set_calls == [(option_key, float(RS_SYNC_MODE_MASTER))]


def test_apply_sync_mode_raises_when_sync_is_required_but_unsupported():
    option_key = object()
    sensor = _StubSensor(name="RGB Camera", supports_sync=False)
    with pytest.raises(RuntimeError, match="inter_cam_sync_mode"):
        apply_sync_mode(
            sensor,
            option_key=option_key,
            role="slave",
            sensor_label="camera_l rgb",
        )


def test_apply_sync_mode_allows_none_role_without_sync_support():
    option_key = object()
    sensor = _StubSensor(name="RGB Camera", supports_sync=False)
    apply_sync_mode(
        sensor,
        option_key=option_key,
        role="none",
        sensor_label="camera_x rgb",
    )
    assert sensor.set_calls == []
