from __future__ import annotations

from robolineage_contracts.pipeline import DatasetLock
from robolineage_dataset.diff import diff_locks


def _lock(version_id: str, rollout_ids: tuple[str, ...]) -> DatasetLock:
    return DatasetLock(
        version_id=version_id,
        created_at="2026-04-25T12:00:00Z",
        base_version_id=None,
        included_rollout_ids=rollout_ids,
        total_samples=len(rollout_ids),
        manifest_sha256="0" * 64,
        changelog="test",
    )


def test_diff_locks():
    diff = diff_locks(_lock("v1", ("r1", "r2")), _lock("v2", ("r2", "r3")))

    assert diff.added_rollout_ids == ("r3",)
    assert diff.removed_rollout_ids == ("r1",)
    assert diff.total_delta == 0
