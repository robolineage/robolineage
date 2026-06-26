from __future__ import annotations

from dataclasses import dataclass

from robolineage_contracts.pipeline import DatasetLock


@dataclass(frozen=True)
class DatasetDiff:
    from_version: str
    to_version: str
    added_rollout_ids: tuple[str, ...]
    removed_rollout_ids: tuple[str, ...]
    total_delta: int


def diff_locks(a: DatasetLock, b: DatasetLock) -> DatasetDiff:
    a_ids = set(a.included_rollout_ids)
    b_ids = set(b.included_rollout_ids)
    return DatasetDiff(
        from_version=a.version_id,
        to_version=b.version_id,
        added_rollout_ids=tuple(sorted(b_ids - a_ids)),
        removed_rollout_ids=tuple(sorted(a_ids - b_ids)),
        total_delta=b.total_samples - a.total_samples,
    )
