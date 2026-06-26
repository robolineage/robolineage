from __future__ import annotations

import os

import pytest
from robolineage_contracts.pipeline import DatasetLock
from robolineage_dataset.lock_writer import DatasetLockWriter


def _lock(version_id: str = "v1") -> DatasetLock:
    return DatasetLock(
        version_id=version_id,
        created_at="2026-04-25T12:00:00Z",
        base_version_id=None,
        included_rollout_ids=("r1",),
        total_samples=1,
        manifest_sha256="0" * 64,
        changelog="test",
    )


def test_write_lock_is_read_only(tmp_path):
    path = DatasetLockWriter().write(_lock(), tmp_path)

    assert path.exists()
    assert not os.access(path, os.W_OK)


def test_write_lock_does_not_overwrite(tmp_path):
    writer = DatasetLockWriter()
    writer.write(_lock(), tmp_path)

    with pytest.raises(FileExistsError):
        writer.write(_lock(), tmp_path)
