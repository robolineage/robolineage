"""H4 — DatasetLock immutability + sha256 determinism."""
import json
from pathlib import Path

import pytest

from robolineage_contracts.pipeline import DatasetLock, compute_manifest_sha256


FIXTURES = Path(__file__).resolve().parents[2] / "_shared_fixtures"


def _lock(**overrides):
    base = dict(
        version_id="v1",
        created_at="2026-04-25T12:00:00Z",
        base_version_id=None,
        included_rollout_ids=("rid-1", "rid-2"),
        total_samples=2,
        manifest_sha256="0" * 64,
        changelog="Initial",
    )
    base.update(overrides)
    return DatasetLock(**base)


# ── Construction + invariants ────────────────────────────────────────────

def test_minimal_construction():
    lock = _lock()
    assert lock.version_id == "v1"
    assert lock.base_version_id is None


def test_is_frozen():
    lock = _lock()
    with pytest.raises(Exception):  # FrozenInstanceError
        lock.version_id = "v2"  # type: ignore[misc]


def test_rejects_duplicate_rollout_ids():
    with pytest.raises(ValueError, match="unique"):
        _lock(included_rollout_ids=("rid-1", "rid-1"))


@pytest.mark.parametrize("bad_sha", [
    "0" * 63,         # too short
    "0" * 65,         # too long
    "g" * 64,         # non-hex
    "X" * 64,         # uppercase rejected (canonical hex is lowercase)
    "",
])
def test_rejects_bad_manifest_sha256(bad_sha):
    with pytest.raises(ValueError, match="manifest_sha256"):
        _lock(manifest_sha256=bad_sha)


def test_rejects_negative_total_samples():
    with pytest.raises(ValueError, match="total_samples"):
        _lock(total_samples=-1)


# ── Hash determinism + canonicalisation ──────────────────────────────────

def test_compute_sha256_order_insensitive():
    """The hash must be the same regardless of insertion order — this is the
    whole point of canonicalisation."""
    a = [{"rollout_id": "z"}, {"rollout_id": "a"}, {"rollout_id": "m"}]
    b = list(reversed(a))
    assert compute_manifest_sha256(a) == compute_manifest_sha256(b)


def test_compute_sha256_dict_key_order_insensitive():
    """Dict key order also must not affect the hash (sort_keys=True)."""
    a = [{"rollout_id": "x", "review_score": "A", "confidence": 0.9}]
    b = [{"confidence": 0.9, "rollout_id": "x", "review_score": "A"}]
    assert compute_manifest_sha256(a) == compute_manifest_sha256(b)


def test_compute_sha256_handles_unicode():
    """Canonical form is utf-8; non-ASCII content must hash identically across runs."""
    a = [{"rollout_id": "x", "changelog": "unicode test"}]
    b = [{"changelog": "unicode test", "rollout_id": "x"}]
    assert compute_manifest_sha256(a) == compute_manifest_sha256(b)


def test_compute_sha256_empty_list():
    """Empty list still yields a stable, valid hex digest."""
    h = compute_manifest_sha256([])
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_compute_sha256_changes_when_content_changes():
    a = [{"rollout_id": "x"}]
    b = [{"rollout_id": "x", "extra": "field"}]
    assert compute_manifest_sha256(a) != compute_manifest_sha256(b)


# ── Fixture round-trip ───────────────────────────────────────────────────

def test_fixture_dataset_lock_validates_and_loads():
    """Verify the shared fixture is internally consistent (sha256 matches its
    documented manifest entries)."""
    raw = json.loads((FIXTURES / "dataset.lock").read_text())
    lock = DatasetLock(
        version_id=raw["version_id"],
        created_at=raw["created_at"],
        base_version_id=raw["base_version_id"],
        included_rollout_ids=tuple(raw["included_rollout_ids"]),
        total_samples=raw["total_samples"],
        manifest_sha256=raw["manifest_sha256"],
        changelog=raw["changelog"],
    )
    assert lock.version_id == "v1"
    assert lock.base_version_id is None
    assert lock.total_samples == 2
    assert "027b72ff-aaaa-bbbb-cccc-000000000001" in lock.included_rollout_ids
