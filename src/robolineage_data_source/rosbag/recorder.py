from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from robolineage_contracts.core import RolloutOutcome


RAW_ROSBAG_MANIFEST_SCHEMA_VERSION = "RoboLineage.raw_rosbag_manifest.v1"


class RosbagRawRecorder:
    """Start and stop a direct ``ros2 bag record`` process for one rollout.

    The recorder intentionally does not subscribe to an in-process transport.
    It lets ROS2 own the hot raw-data path and stores only a small manifest
    next to the bag so downstream dataset adapters can find the captured
    topics.
    """

    def __init__(
        self,
        *,
        rollout_dir: str | Path,
        topics: tuple[str, ...] | list[str],
        ros_domain_id: int | None = None,
        storage_id: str | None = None,
        process_factory: Callable[..., Any] | None = None,
    ) -> None:
        cleaned = tuple(dict.fromkeys(str(topic).strip() for topic in topics if str(topic).strip()))
        if not cleaned:
            raise ValueError("RosbagRawRecorder requires at least one ROS topic")
        self.rollout_dir = Path(rollout_dir)
        self.raw_dir = self.rollout_dir / "raw"
        self.bag_dir = self.raw_dir / "rosbag2"
        self.manifest_path = self.raw_dir / "raw_manifest.json"
        self.topics = cleaned
        self.ros_domain_id = ros_domain_id
        self.storage_id = str(storage_id).strip() if storage_id else None
        self._process_factory = process_factory or subprocess.Popen
        self._process: Any | None = None
        self._started = False
        self._capturing = False
        self._finalized = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("RosbagRawRecorder already started")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["ros2", "bag", "record"]
        if self.storage_id:
            cmd.extend(["--storage", self.storage_id])
        cmd.extend(["--output", str(self.bag_dir)])
        cmd.extend(self.topics)
        env = os.environ.copy()
        if self.ros_domain_id is not None:
            env["ROS_DOMAIN_ID"] = str(int(self.ros_domain_id))
        self._write_manifest(
            {
                "status": "recording",
                "started_at": _now_iso(),
                "returncode": None,
                "outcome": None,
                "note": None,
                "command": cmd,
            }
        )
        self._process = self._process_factory(cmd, env=env)
        self._started = True
        self._capturing = True
        self._finalized = False

    def stop_capture(self) -> None:
        if not self._started or not self._capturing:
            return
        process = self._process
        returncode: int | None = None
        if process is not None:
            try:
                process.terminate()
                returncode = int(process.wait(timeout=10.0))
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = int(process.wait(timeout=5.0))
        self._capturing = False
        self._write_manifest(
            {
                "status": "capture_stopped",
                "stopped_at": _now_iso(),
                "returncode": returncode,
            }
        )

    def finalize(
        self,
        *,
        outcome: RolloutOutcome = RolloutOutcome.SUCCESS,
        note: str = "closed by orchestrator",
    ) -> None:
        if not self._started or self._finalized:
            return
        self.stop_capture()
        returncode = None
        if self._process is not None and self._process.returncode is not None:
            returncode = int(self._process.returncode)
        self._write_manifest(
            {
                "status": "closed",
                "closed_at": _now_iso(),
                "outcome": outcome.value if hasattr(outcome, "value") else str(outcome),
                "note": note,
                "returncode": returncode,
            }
        )
        self._finalized = True
        self._started = False

    def stop(
        self,
        *,
        outcome: RolloutOutcome = RolloutOutcome.SUCCESS,
        note: str = "closed by orchestrator",
    ) -> None:
        self.stop_capture()
        self.finalize(outcome=outcome, note=note)

    def _write_manifest(self, patch: dict[str, Any]) -> None:
        payload: dict[str, Any] = {}
        if self.manifest_path.exists():
            try:
                payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        payload.update(
            {
                "schema_version": RAW_ROSBAG_MANIFEST_SCHEMA_VERSION,
                "recorder": "ros2_bag_record",
                "raw_format": "rosbag2",
                "rollout_dir": str(self.rollout_dir),
                "raw_dir": str(self.raw_dir),
                "bag_dir": str(self.bag_dir),
                "topics": list(self.topics),
                "ros_domain_id": self.ros_domain_id,
                "storage_id": self.storage_id,
            }
        )
        payload.update(patch)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.manifest_path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
