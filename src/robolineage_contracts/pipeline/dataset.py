"""DatasetLock — immutable freeze-point for a dataset version.

Source of truth: docs/artifact_contracts.md §2 (constraint 4) +
docs/artifact_contracts.md §13.

Producer: current DatasetUpdater (`src/robolineage_dataset/updater.py`) or
TrainingLifecycleRunner. Consumer: `robolineage_train` and check/deployment tooling
(must NEVER write a different `trained_on_dataset` than what its
DatasetLock-derived input says — see PolicyMeta).

Contract: once written, a dataset.lock file is **read-only**. Any change to
the dataset (add / remove rollouts) produces a new version with a new lock.
The writer chmods the file 0o444 after the atomic rename.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DatasetLock:
    """The immutable record of a dataset version's contents."""
    version_id: str                       # "v1", "v2", ... monotonic int suffix
    created_at: str                       # ISO8601
    base_version_id: str | None           # parent version; None for v1
    included_rollout_ids: Tuple[str, ...]  # unique, order is sorted-by-rollout_id
    total_samples: int
    manifest_sha256: str                  # hex digest of the canonical manifest
    changelog: str                        # human-readable summary

    def __post_init__(self) -> None:
        # Uniqueness — duplicates are a dataset-builder bug, surface immediately.
        if len(set(self.included_rollout_ids)) != len(self.included_rollout_ids):
            raise ValueError(
                "included_rollout_ids must be unique; "
                f"got {len(self.included_rollout_ids)} entries with "
                f"{len(set(self.included_rollout_ids))} unique"
            )
        # SHA256 hex is 64 chars — anything else is wrong.
        if len(self.manifest_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.manifest_sha256
        ):
            raise ValueError(
                f"manifest_sha256 must be 64 hex chars; got {self.manifest_sha256!r}"
            )
        if self.total_samples < 0:
            raise ValueError(f"total_samples must be ≥0; got {self.total_samples}")


@dataclass(frozen=True)
class DatasetVersion:
    """High-level dataset version metadata (companion to DatasetLock).

    DatasetLock is the *content fingerprint*; DatasetVersion is the
    *human-facing description* (counts by bucket, source, etc.). Some
    deployments embed DatasetVersion fields inside dataset.lock; RoboLineage
    keeps them separate so the lock stays minimal + content-addressed.
    """
    version_id: str
    created_at: str
    base_version_id: str | None
    total_samples: int
    train_count: int
    review_count: int
    source_rollout_ids: Tuple[str, ...]   # newly added in this version (delta from base)
    changelog: str

    def __post_init__(self) -> None:
        for label, value in (
            ("total_samples", self.total_samples),
            ("train_count", self.train_count),
            ("review_count", self.review_count),
        ):
            if value < 0:
                raise ValueError(f"{label} must be ≥0; got {value}")


# ── Canonical manifest hashing ────────────────────────────────────────────
#
# Dataset writers and PolicyMetaWriter.verify_integrity call into this
# function. Re-implementing the hash anywhere else is a contract violation.

def compute_manifest_sha256(manifest_entries: list[dict]) -> str:
    """SHA256 of the canonical-form manifest.

    Canonicalisation:
      1. Sort entries by `rollout_id` (stable; missing keys treated as "")
      2. JSON-serialise with `sort_keys=True, ensure_ascii=False`
      3. UTF-8 encode

    Stable across:
      - Insertion order of entries  → step 1
      - Dict key order              → step 2 (sort_keys)
      - Locale / Python build       → step 3 (utf-8)
    """
    sorted_entries = sorted(manifest_entries, key=lambda e: e.get("rollout_id", ""))
    canonical_bytes = json.dumps(
        sorted_entries, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()
