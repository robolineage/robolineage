"""Phase 1 RosArxOneAdapter unit tests.

The pure callback tests run without ROS2. The real rclpy start/stop smoke test
auto-skips on machines that do not have ROS2 installed.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest


def _has_rclpy() -> bool:
    try:
        import rclpy  # noqa: F401

        return True
    except Exception:
        return False


def _make_fake_compressed_image(
    jpeg_bytes: bytes,
    header_sec: int = 1,
    header_nanosec: int = 0,
):
    msg = MagicMock()
    msg.data = jpeg_bytes
    msg.header.stamp.sec = header_sec
    msg.header.stamp.nanosec = header_nanosec
    return msg


def _make_fake_robot_status():
    msg = MagicMock()
    msg.joint_pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.5]
    msg.joint_vel = [0.0] * 7
    msg.joint_cur = [0.0] * 7
    msg.end_pos = [0.5, 0.0, 0.3, 0.0, 0.0, 0.0]
    msg.header.stamp.sec = 100
    msg.header.stamp.nanosec = 500_000_000
    return msg


def _make_jpeg_bytes(width: int = 8, height: int = 8) -> bytes:
    import cv2

    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return bytes(encoded)


def test_robot_status_to_27_vec_layout():
    from robolineage_data_source.adapters.ros2_arx_one import _robot_status_to_27_vec

    msg = Mock(
        joint_pos=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        joint_vel=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        joint_cur=[2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
        end_pos=[10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
    )
    vec = _robot_status_to_27_vec(msg)
    assert vec.shape == (27,)
    assert vec.dtype == np.float32
    assert np.isclose(vec[0], 0.1)
    assert np.isclose(vec[6], 0.7)
    assert np.isclose(vec[7], 1.0)
    assert np.isclose(vec[14], 2.0)
    assert np.isclose(vec[21], 10.0)
    assert np.isclose(vec[26], 0.3)


def test_on_camera_updates_health_and_frame_index_without_transport_publish():
    from robolineage_data_source.adapters.ros2_arx_one import RosArxOneAdapter
    from robolineage_data_source.config.schema import CameraTopicSpec, Ros2AdapterConfig

    cam_spec = CameraTopicSpec(
        topic="/camera/camera_h/color/image_raw/compressed",
        transport="compressed",
        qos="sensor_data",
        stream_id="cam/camera_h/color",
    )
    cfg = Ros2AdapterConfig(cameras={"camera_h": cam_spec})
    adapter = RosArxOneAdapter(config=cfg)
    adapter._frame_indices[cam_spec.stream_id] = 0

    msg = _make_fake_compressed_image(_make_jpeg_bytes())
    adapter._on_camera(msg, cam_spec)

    health = adapter.health()
    assert health.last_sample_mono_ns is not None
    assert health.meta["last_camera_topic"] == "/camera/camera_h/color/image_raw/compressed"
    assert health.meta["last_camera_frame_index"] == 0
    assert health.meta["last_camera_stamp_ns"] == 1_000_000_000

    adapter._on_camera(msg, cam_spec)
    assert adapter.health().meta["last_camera_frame_index"] == 1


def test_on_camera_exposes_latest_decoded_frame_for_preview():
    from robolineage_data_source.adapters.ros2_arx_one import RosArxOneAdapter
    from robolineage_data_source.config.schema import CameraTopicSpec, Ros2AdapterConfig

    cam_spec = CameraTopicSpec(
        topic="/camera/camera_h/color/image_raw/compressed",
        transport="compressed",
        qos="sensor_data",
        stream_id="cam/camera_h/color",
    )
    cfg = Ros2AdapterConfig(cameras={"camera_h": cam_spec})
    adapter = RosArxOneAdapter(config=cfg)
    adapter._frame_indices[cam_spec.stream_id] = 0

    adapter._on_camera(_make_fake_compressed_image(_make_jpeg_bytes(width=12, height=10)), cam_spec)

    frame = adapter.latest_camera_frame(stream_id=cam_spec.stream_id)
    assert frame is not None
    assert frame.shape == (10, 12, 3)
    status = adapter.camera_status(stream_id=cam_spec.stream_id)
    assert status is not None
    assert status["topic"] == "/camera/camera_h/color/image_raw/compressed"
    assert status["shape"] == [10, 12, 3]


def test_on_arm_pose_updates_health_with_27_vec_metadata():
    from robolineage_data_source.adapters.ros2_arx_one import RosArxOneAdapter
    from robolineage_data_source.config.schema import ArmTopicSpec, Ros2AdapterConfig

    arm_spec = ArmTopicSpec(
        slave_status="/arm_slave_l_status",
        state_stream_id="robot/left_arm/state",
        qos="reliable",
    )
    cfg = Ros2AdapterConfig(arms={"left_arm": arm_spec})
    adapter = RosArxOneAdapter(config=cfg)

    msg = _make_fake_robot_status()
    adapter._on_arm_pose(msg, arm_spec, "left_arm")

    health = adapter.health()
    assert health.last_sample_mono_ns is not None
    assert health.meta["last_arm"] == "left_arm"
    assert health.meta["last_arm_topic"] == "/arm_slave_l_status"
    assert health.meta["last_arm_stamp_ns"] == 100_500_000_000
    assert health.meta["last_arm_vector_len"] == 27


def test_on_master_command_updates_health_without_overlay_transport():
    from robolineage_data_source.adapters.ros2_arx_one import RosArxOneAdapter
    from robolineage_data_source.config.schema import ArmTopicSpec, Ros2AdapterConfig

    arm_spec = ArmTopicSpec(
        slave_status="/arm_slave_l_status",
        state_stream_id="robot/left_arm/state",
        qos="reliable",
        master_command="/arm_master_l_status",
    )
    cfg = Ros2AdapterConfig(
        arms={"left_arm": arm_spec},
        master_overlay_topic="robot/active/command",
    )
    adapter = RosArxOneAdapter(config=cfg)

    msg = _make_fake_robot_status()
    adapter._on_master_command(msg, "left_arm")

    health = adapter.health()
    assert health.meta["last_master_arm"] == "left_arm"
    assert health.meta["last_master_stamp_ns"] == 100_500_000_000
    assert health.meta["last_master_vector_len"] == 27


def test_on_camera_jpeg_decode_failure_marks_degraded_without_publish():
    from robolineage_data_source.adapters.ros2_arx_one import RosArxOneAdapter
    from robolineage_data_source.config.schema import CameraTopicSpec, Ros2AdapterConfig
    from robolineage_data_source.sample import HealthState

    cam_spec = CameraTopicSpec(
        topic="/camera/camera_h/color/image_raw/compressed",
        transport="compressed",
        qos="sensor_data",
        stream_id="cam/camera_h/color",
    )
    cfg = Ros2AdapterConfig(cameras={"camera_h": cam_spec})
    adapter = RosArxOneAdapter(config=cfg)
    adapter._frame_indices[cam_spec.stream_id] = 0

    bad_msg = _make_fake_compressed_image(b"not a jpeg")
    adapter._on_camera(bad_msg, cam_spec)

    assert adapter.health().state == HealthState.DEGRADED


def test_jpeg_decode_failures_counted_per_stream_and_surfaced_via_health_meta():
    """Phase 5 T3: per-stream failure counter visible through health().meta
    so the /health endpoint can show degradation depth, not just a binary state."""
    from robolineage_data_source.adapters.ros2_arx_one import RosArxOneAdapter
    from robolineage_data_source.config.schema import CameraTopicSpec, Ros2AdapterConfig

    cam_h = CameraTopicSpec(
        topic="/h", transport="compressed", qos="sensor_data",
        stream_id="cam/camera_h/color",
    )
    cam_l = CameraTopicSpec(
        topic="/l", transport="compressed", qos="sensor_data",
        stream_id="cam/camera_l/color",
    )
    cfg = Ros2AdapterConfig(cameras={"camera_h": cam_h, "camera_l": cam_l})
    adapter = RosArxOneAdapter(config=cfg)
    adapter._frame_indices[cam_h.stream_id] = 0
    adapter._frame_indices[cam_l.stream_id] = 0

    # Fresh adapter — counters all zero
    assert adapter.health().meta["jpeg_decode_failures"] == {}

    bad = _make_fake_compressed_image(b"not jpeg")
    adapter._on_camera(bad, cam_h)
    adapter._on_camera(bad, cam_h)
    adapter._on_camera(bad, cam_l)

    failures = adapter.health().meta["jpeg_decode_failures"]
    assert failures["cam/camera_h/color"] == 2
    assert failures["cam/camera_l/color"] == 1

    # Successful decodes do NOT increment counter
    good = _make_fake_compressed_image(_make_jpeg_bytes())
    adapter._on_camera(good, cam_h)
    failures = adapter.health().meta["jpeg_decode_failures"]
    assert failures["cam/camera_h/color"] == 2  # still 2, not 3


@pytest.mark.skipif(not _has_rclpy(), reason="rclpy not installed (Linux + ROS2 only)")
def test_start_and_stop_smoke():
    from robolineage_data_source.adapters.ros2_arx_one import RosArxOneAdapter
    from robolineage_data_source.config.schema import Ros2AdapterConfig

    cfg = Ros2AdapterConfig()
    adapter = RosArxOneAdapter(config=cfg)
    adapter.start()
    try:
        time.sleep(0.5)
        assert adapter.health().state.value == "ok"
    finally:
        adapter.stop()
