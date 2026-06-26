from __future__ import annotations

import json
from pathlib import Path

from robolineage_contracts.core import RolloutOutcome

from robolineage_data_source.rosbag.recorder import RosbagRawRecorder


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0 if self.returncode is None else self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_rosbag_raw_recorder_starts_ros2_bag_record_and_writes_manifest(tmp_path):
    calls: list[list[str]] = []
    process = _FakeProcess()

    def factory(cmd: list[str], **kwargs):
        calls.append(cmd)
        return process

    recorder = RosbagRawRecorder(
        rollout_dir=tmp_path / "rollouts" / "r1",
        topics=(
            "/cam/head/image/compressed",
            "/cam/right_wrist/image/compressed",
            "/arm/left/state",
            "/arm/right/state",
        ),
        ros_domain_id=23,
        storage_id="mcap",
        process_factory=factory,
    )

    recorder.start()

    raw_dir = tmp_path / "rollouts" / "r1" / "raw"
    manifest_path = raw_dir / "raw_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert calls == [
        [
            "ros2",
            "bag",
            "record",
            "--storage",
            "mcap",
            "--output",
            str(raw_dir / "rosbag2"),
            "/cam/head/image/compressed",
            "/cam/right_wrist/image/compressed",
            "/arm/left/state",
            "/arm/right/state",
        ]
    ]
    assert manifest["schema_version"] == "RoboLineage.raw_rosbag_manifest.v1"
    assert manifest["status"] == "recording"
    assert manifest["ros_domain_id"] == 23
    assert manifest["topics"] == [
        "/cam/head/image/compressed",
        "/cam/right_wrist/image/compressed",
        "/arm/left/state",
        "/arm/right/state",
    ]
    assert manifest["bag_dir"] == str(raw_dir / "rosbag2")

    recorder.stop_capture()
    recorder.finalize(outcome=RolloutOutcome.SUCCESS, note="test closed")

    closed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert process.terminated is True
    assert closed["status"] == "closed"
    assert closed["outcome"] == "success"
    assert closed["note"] == "test closed"
    assert closed["returncode"] == 0
