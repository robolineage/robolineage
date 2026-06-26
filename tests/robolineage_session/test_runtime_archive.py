from pathlib import Path

import pytest

from robolineage_session.runtime_archive import archive


def test_archive_moves_runtime_snapshots(tmp_path: Path):
    runtime = tmp_path / "runtime"
    rollout = tmp_path / "rollout"
    runtime.mkdir()
    (runtime / "snapshots.jsonl").write_text("one\n", encoding="utf-8")

    target = archive(runtime, rollout)

    assert target == rollout / "snapshots.jsonl"
    assert target.read_text(encoding="utf-8") == "one\n"
    assert not (runtime / "snapshots.jsonl").exists()


def test_archive_rejects_existing_target(tmp_path: Path):
    runtime = tmp_path / "runtime"
    rollout = tmp_path / "rollout"
    runtime.mkdir()
    rollout.mkdir()
    (runtime / "snapshots.jsonl").write_text("one\n", encoding="utf-8")
    (rollout / "snapshots.jsonl").write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        archive(runtime, rollout)


def test_archive_creates_empty_target_for_empty_runtime(tmp_path: Path):
    target = archive(tmp_path / "runtime", tmp_path / "rollout")

    assert target.exists()
    assert target.read_text(encoding="utf-8") == ""
