import json

import pytest

from robolineage_contracts.agents import SnapshotAssessment, SnapshotTrigger
from robolineage_shared_agents.visual_snapshot.output import SnapshotWriter


def _snapshot(frame_id: int = 1) -> SnapshotAssessment:
    return SnapshotAssessment(
        timestamp=frame_id * 0.1,
        frame_id=frame_id,
        progress="advancing",
        risk_level="low",
        phase="grasp",
        imminent_failure=False,
        confidence=0.9,
        needs_review=False,
        raw_response="{}",
        trigger=SnapshotTrigger.GRIPPER_CLOSE,
        frame_index_range=(frame_id, frame_id),
        vlm_meta={"model": "mock", "latency_ms": 1, "prompt_version": "inline-v1"},
    )


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    with SnapshotWriter(path) as writer:
        writer.write(_snapshot())

    line = json.loads(path.read_text(encoding="utf-8"))
    reloaded = SnapshotAssessment(**line)
    assert reloaded.frame_id == 1
    assert reloaded.vlm_meta["model"] == "mock"


def test_two_live_writers_for_same_path_fail(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    first = SnapshotWriter(path)
    try:
        with pytest.raises(FileExistsError):
            SnapshotWriter(path)
    finally:
        first.close()


def test_close_then_reopen_appends(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    with SnapshotWriter(path) as writer:
        writer.write(_snapshot(1))

    with SnapshotWriter(path) as writer:
        writer.write(_snapshot(2))

    rows = [SnapshotAssessment(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row.frame_id for row in rows] == [1, 2]
