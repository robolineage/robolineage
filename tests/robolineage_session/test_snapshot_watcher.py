import json
from pathlib import Path

from robolineage_session.snapshot_watcher import SnapshotWatcher


def _snapshot(frame_id: int) -> dict:
    return {
        "timestamp": float(frame_id),
        "frame_id": frame_id,
        "progress": "advancing",
        "risk_level": "low",
        "phase": "approach",
        "imminent_failure": False,
        "confidence": 0.9,
        "needs_review": False,
        "raw_response": "ok",
    }


def test_watcher_reads_new_snapshot_lines(tmp_path: Path):
    path = tmp_path / "snapshots.jsonl"
    path.write_text(
        "\n".join(json.dumps(_snapshot(i)) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    watcher = SnapshotWatcher(path, poll_interval=0.01)
    gen = watcher.iter_new()

    frames = [next(gen).frame_id for _ in range(3)]
    watcher.stop()

    assert frames == [0, 1, 2]


def test_watcher_skips_broken_complete_line(tmp_path: Path):
    path = tmp_path / "snapshots.jsonl"
    path.write_text("{broken json}\n" + json.dumps(_snapshot(7)) + "\n", encoding="utf-8")
    watcher = SnapshotWatcher(path, poll_interval=0.01)
    gen = watcher.iter_new()

    snapshot = next(gen)
    watcher.stop()

    assert snapshot.frame_id == 7
