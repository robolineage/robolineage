"""Tests for config loader."""
from pathlib import Path

import pytest

from robolineage_data_source.config.loader import load_config
from robolineage_data_source.config.schema import Config


YAML_MINIMAL = """
rollout:
  task_id: task_98
  operator_id: op_001
"""

YAML_FULL = """
rollout:
  task_id: task_98
  operator_id: op_001

sync_groups:
  - name: main
    backend: realsense_inter_cam
    master: camera_h
    slaves: [camera_l, camera_r]

cameras:
  camera_h:
    type: realsense
    serial: "135122073233"
    resolution: [1280, 720]
    fps: 30
    depth: true
  camera_l:
    type: realsense
    serial: "129122070025"
  camera_r:
    type: realsense
    serial: "135122074281"

imu:
  main:
    type: serial
    port: /dev/ttyUSB0
    rate: 200

robots:
  arx_left:
    type: arx_one
    poll_rate: 200
    can_bus: can1

recorder:
  output_dir: data/rollouts

preview:
  bind: 0.0.0.0:8080
  stream_bitrate: 5000000
"""


def write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


def test_load_minimal(tmp_path):
    p = write_yaml(tmp_path, YAML_MINIMAL)
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.rollout.task_id == "task_98"
    assert cfg.rollout.operator_id == "op_001"
    assert cfg.rollout.mode == "C1"
    assert cfg.rollout.policy_version is None
    assert cfg.sync_groups == []
    assert cfg.cameras == {}
    assert cfg.recorder is None
    assert cfg.preview is None


def test_load_full(tmp_path):
    p = write_yaml(tmp_path, YAML_FULL)
    cfg = load_config(p)

    # rollout
    assert cfg.rollout.task_id == "task_98"

    # sync_groups
    assert len(cfg.sync_groups) == 1
    g = cfg.sync_groups[0]
    assert g.name == "main"
    assert g.backend == "realsense_inter_cam"
    assert g.master == "camera_h"
    assert g.slaves == ["camera_l", "camera_r"]

    # cameras
    assert set(cfg.cameras.keys()) == {"camera_h", "camera_l", "camera_r"}
    h = cfg.cameras["camera_h"]
    assert h.type == "realsense"
    assert h.serial == "135122073233"
    assert h.resolution == (1280, 720)
    assert h.fps == 30
    assert h.depth is True
    l = cfg.cameras["camera_l"]
    # defaults applied
    assert l.resolution == (1280, 720)
    assert l.fps == 30
    assert l.depth is False

    # imu
    assert cfg.imu["main"].port == "/dev/ttyUSB0"
    assert cfg.imu["main"].rate == 200

    # robots
    r = cfg.robots["arx_left"]
    assert r.type == "arx_one"
    assert r.poll_rate == 200
    assert r.extra == {"can_bus": "can1"}

    # recorder
    assert cfg.recorder.output_dir == "data/rollouts"
    assert cfg.recorder.camera_names is None

    # preview
    assert cfg.preview.bind == "0.0.0.0:8080"
    assert cfg.preview.stream_bitrate == 5_000_000


def test_load_recorder_camera_names(tmp_path):
    p = write_yaml(
        tmp_path,
        """
rollout:
  task_id: task_98
  operator_id: op_001
recorder:
  output_dir: data/rollouts
  camera_names: [camera_h, camera_r]
""",
    )
    cfg = load_config(p)
    assert cfg.recorder is not None
    assert cfg.recorder.camera_names == ("camera_h", "camera_r")


def test_load_rejects_non_mapping(tmp_path):
    p = write_yaml(tmp_path, "- just a list\n- of things\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_load_rejects_missing_rollout(tmp_path):
    p = write_yaml(tmp_path, "cameras: {}\n")
    with pytest.raises(ValueError, match="rollout"):
        load_config(p)


def test_load_accepts_pathlike(tmp_path):
    p = write_yaml(tmp_path, YAML_MINIMAL)
    cfg = load_config(str(p))
    assert cfg.rollout.task_id == "task_98"


def test_extra_camera_keys_are_preserved(tmp_path):
    yaml_text = """
rollout:
  task_id: t
  operator_id: o
cameras:
  c1:
    type: gopro
    ip: "203.0.113.50"
    token: camera-token-placeholder
"""
    p = write_yaml(tmp_path, yaml_text)
    cfg = load_config(p)
    assert cfg.cameras["c1"].type == "gopro"
    assert cfg.cameras["c1"].extra == {"ip": "203.0.113.50", "token": "camera-token-placeholder"}
