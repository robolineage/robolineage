from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from robolineage_contracts.pipeline import DatasetLock


def make_dataset_lock(version_id: str = "v1") -> DatasetLock:
    return DatasetLock(
        version_id=version_id,
        created_at="2026-04-25T12:00:00Z",
        base_version_id=None,
        included_rollout_ids=("rollout1",),
        total_samples=1,
        manifest_sha256="0" * 64,
        changelog="test dataset",
    )


def write_dataset_lock(path: Path, version_id: str = "v1") -> DatasetLock:
    lock = make_dataset_lock(version_id)
    path.write_text(json.dumps(asdict(lock), ensure_ascii=False), encoding="utf-8")
    return lock


def write_report(path: Path) -> Path:
    path.write_text('{"rollout_id":"rollout1","result":"reviewed"}\n', encoding="utf-8")
    return path
