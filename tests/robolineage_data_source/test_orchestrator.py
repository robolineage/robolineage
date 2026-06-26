"""Tests for Orchestrator — uses MockAdapter via a test-only factory, no hardware."""
import time
from pathlib import Path

import pytest

from robolineage_data_source.adapters.mock import MockAdapter
from robolineage_data_source.config.loader import load_config
from robolineage_data_source.orchestrator import Orchestrator


@pytest.fixture
def mock_config(tmp_path):
    yaml = f"""
rollout:
  task_id: t1
  operator_id: op
cameras:
  cam_a:
    type: mock
  cam_b:
    type: mock
recorder:
  output_dir: {tmp_path}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml)
    return p


def _mock_adapter_factory(name: str, spec: dict):
    if spec.get("type") == "mock":
        return MockAdapter(topic=f"cam/{name}/color", rate_hz=50.0)
    raise ValueError(f"no factory for {spec!r}")


def test_orchestrator_starts_and_stops_sources(mock_config):
    cfg = load_config(mock_config)
    orch = Orchestrator(cfg, adapter_factory=_mock_adapter_factory,
                        rollout_id="rollout_001")
    orch.start()
    time.sleep(0.2)
    orch.stop()
    assert orch._started is False


def test_orchestrator_stop_before_start_is_noop(mock_config):
    cfg = load_config(mock_config)
    orch = Orchestrator(cfg, adapter_factory=_mock_adapter_factory,
                        rollout_id="r")
    orch.stop()


def test_orchestrator_rollout_id_is_generated_if_none(mock_config, tmp_path):
    cfg = load_config(mock_config)
    orch = Orchestrator(cfg, adapter_factory=_mock_adapter_factory)
    assert orch.rollout_id
    assert len(orch.rollout_id) >= 8


def test_orchestrator_none_mode_constructs_no_recorders(mock_config):
    cfg = load_config(mock_config)
    orch = Orchestrator(
        cfg,
        adapter_factory=_mock_adapter_factory,
        rollout_id="rollout_001",
        recorder_mode="none",
    )
    assert orch._recorder is None


def test_orchestrator_rejects_unknown_recorder_mode(mock_config):
    cfg = load_config(mock_config)
    with pytest.raises(ValueError, match="recorder_mode"):
        Orchestrator(cfg, adapter_factory=_mock_adapter_factory, recorder_mode="stream")


def test_orchestrator_rosbag_mode_constructs_rosbag_recorder(tmp_path):
    from robolineage_data_source.config.schema import (
        ArmTopicSpec,
        CameraTopicSpec,
        Config,
        RecorderConfig,
        RolloutConfig,
        Ros2AdapterConfig,
    )

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="o"),
        recorder=RecorderConfig(output_dir=str(tmp_path)),
        adapter=Ros2AdapterConfig(
            cameras={
                "head": CameraTopicSpec(
                    topic="/cam/head/image/compressed",
                    transport="compressed",
                    qos="sensor_data",
                    stream_id="cam/head/color",
                )
            },
            arms={
                "left": ArmTopicSpec(
                    slave_status="/arm/left/state",
                    state_stream_id="robot/left/state",
                    msg_type="example_msgs/msg/RobotState",
                )
            },
        ),
    )
    orch = Orchestrator(cfg, rollout_id="r1", recorder_mode="rosbag")
    assert type(orch._recorder).__name__ == "RosbagRawRecorder"
    assert orch._recorder.topics == ("/cam/head/image/compressed", "/arm/left/state")


def test_orchestrator_rosbag_topics_respect_recorder_camera_names(tmp_path):
    from robolineage_data_source.config.schema import (
        ArmTopicSpec,
        CameraTopicSpec,
        Config,
        RecorderConfig,
        RolloutConfig,
        Ros2AdapterConfig,
    )

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="o"),
        recorder=RecorderConfig(
            output_dir=str(tmp_path),
            camera_names=("camera_h", "camera_r"),
        ),
        adapter=Ros2AdapterConfig(
            cameras={
                "camera_h": CameraTopicSpec(
                    topic="/camera/camera_h/color/image_raw/compressed",
                    transport="compressed",
                    qos="sensor_data",
                    stream_id="cam/camera_h/color",
                    camera_name="camera_h",
                ),
                "camera_l": CameraTopicSpec(
                    topic="/camera/camera_l/color/image_raw/compressed",
                    transport="compressed",
                    qos="sensor_data",
                    stream_id="cam/camera_l/color",
                    camera_name="camera_l",
                ),
                "camera_r": CameraTopicSpec(
                    topic="/camera/camera_r/color/image_raw/compressed",
                    transport="compressed",
                    qos="sensor_data",
                    stream_id="cam/camera_r/color",
                    camera_name="camera_r",
                ),
            },
            arms={
                "left": ArmTopicSpec(
                    slave_status="/arm_master_l_status",
                    state_stream_id="robot/left/state",
                    msg_type="example_msgs/msg/RobotState",
                ),
                "right": ArmTopicSpec(
                    slave_status="/arm_master_r_status",
                    state_stream_id="robot/right/state",
                    msg_type="example_msgs/msg/RobotState",
                ),
            },
        ),
    )
    orch = Orchestrator(cfg, rollout_id="r1", recorder_mode="rosbag")

    assert orch._recorder.topics == (
        "/camera/camera_h/color/image_raw/compressed",
        "/camera/camera_r/color/image_raw/compressed",
        "/arm_master_l_status",
        "/arm_master_r_status",
    )


def test_orchestrator_uses_ros2_adapter_when_adapter_section_present():
    from robolineage_data_source.config.schema import (
        ArmTopicSpec,
        CameraTopicSpec,
        Config,
        RolloutConfig,
        Ros2AdapterConfig,
    )

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="o"),
        adapter=Ros2AdapterConfig(
            cameras={
                "camera_h": CameraTopicSpec(
                    topic="/c/h",
                    transport="compressed",
                    qos="sensor_data",
                    stream_id="cam/camera_h/color",
                )
            },
            arms={
                "left_arm": ArmTopicSpec(
                    slave_status="/arm_slave_l_status",
                    state_stream_id="robot/left_arm/state",
                    qos="reliable",
                    msg_type="example_msgs/msg/RobotState",
                )
            },
        ),
    )
    orch = Orchestrator(cfg)
    assert len(orch._adapters) == 1
    name, adapter = next(iter(orch._adapters.items()))
    assert name == "ros2_profile"
    assert type(adapter).__name__ == "Ros2ProfileAdapter"


def test_orchestrator_rejects_unknown_top_level_adapter_type():
    from robolineage_data_source.config.schema import Config, RolloutConfig, Ros2AdapterConfig

    cfg = Config(
        rollout=RolloutConfig(task_id="t", operator_id="o"),
        adapter=Ros2AdapterConfig(type="other"),
    )
    with pytest.raises(ValueError, match="adapter.type"):
        Orchestrator(cfg)


from robolineage_data_source.orchestrator import default_adapter_factory


def test_default_adapter_factory_mock():
    adapter = default_adapter_factory("cam_a", {"type": "mock", "rate_hz": 10.0})
    assert isinstance(adapter, MockAdapter)


def test_default_adapter_factory_unknown_raises():
    with pytest.raises(ValueError, match="unknown adapter type"):
        default_adapter_factory("x", {"type": "foo"})


def test_orchestrator_stop_continues_when_recorder_raises(mock_config, tmp_path):
    """A recorder.stop() exception must not skip later cleanup or crash."""
    from unittest.mock import MagicMock

    cfg = load_config(mock_config)
    orch = Orchestrator(
        cfg,
        adapter_factory=_mock_adapter_factory,
        rollout_id="rollout_stop_test",
        recorder_mode="none",
    )
    orch._recorder = MagicMock()
    orch.start()
    time.sleep(0.05)

    bad_recorder = MagicMock(side_effect=RuntimeError("recorder boom"))
    orch._recorder.stop = bad_recorder

    # Should NOT raise.
    orch.stop()

    bad_recorder.assert_called_once()
    assert orch._started is False
