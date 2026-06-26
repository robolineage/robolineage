from __future__ import annotations

import json

import pytest
from robolineage_contracts.pipeline import compute_manifest_sha256
from robolineage_dataset.updater import DatasetUpdater


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _entry(rollout_id: str) -> dict:
    return {
        "export_id": rollout_id,
        "rollout_id": rollout_id,
        "sample_dir": f"rollouts/{rollout_id}",
        "review_score": "A",
        "confidence": 0.9,
        "l1_phases": None,
        "reasons": ["accepted"],
    }


def test_update_creates_first_version(tmp_path):
    manifest = tmp_path / "train_manifest.jsonl"
    _write_jsonl(manifest, [_entry("r1")])

    lock = DatasetUpdater().update(train_manifest_path=manifest, prev_lock_path=None, out_dir=tmp_path / "datasets")

    assert lock.version_id == "v1"
    assert lock.base_version_id is None
    assert lock.included_rollout_ids == ("r1",)
    assert (tmp_path / "datasets" / "v1" / "dataset.lock").exists()
    assert (tmp_path / "datasets" / "v1" / "manifest.jsonl").exists()


def test_update_merges_previous_manifest_and_deduplicates(tmp_path):
    datasets = tmp_path / "datasets"
    manifest_v1 = tmp_path / "train_manifest_v1.jsonl"
    manifest_v2 = tmp_path / "train_manifest_v2.jsonl"
    _write_jsonl(manifest_v1, [_entry("r1")])
    v1 = DatasetUpdater().update(train_manifest_path=manifest_v1, prev_lock_path=None, out_dir=datasets)
    _write_jsonl(manifest_v2, [_entry("r2"), _entry("r1")])

    v2 = DatasetUpdater().update(
        train_manifest_path=manifest_v2,
        prev_lock_path=datasets / v1.version_id / "dataset.lock",
        out_dir=datasets,
    )

    assert v2.version_id == "v2"
    assert v2.base_version_id == "v1"
    assert v2.included_rollout_ids == ("r1", "r2")
    rows = [json.loads(line) for line in (datasets / "v2" / "manifest.jsonl").read_text().splitlines()]
    assert v2.manifest_sha256 == compute_manifest_sha256(rows)


def test_update_refuses_existing_lock(tmp_path):
    manifest = tmp_path / "train_manifest.jsonl"
    _write_jsonl(manifest, [_entry("r1")])
    DatasetUpdater().update(train_manifest_path=manifest, prev_lock_path=None, out_dir=tmp_path / "datasets")

    with pytest.raises(FileExistsError):
        DatasetUpdater().update(train_manifest_path=manifest, prev_lock_path=None, out_dir=tmp_path / "datasets")
