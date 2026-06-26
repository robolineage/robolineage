"""check_contracts_imports.py: cross-domain import detection + whitelist."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_contracts_imports  # noqa: E402
from check_contracts_imports import (  # noqa: E402
    Violation,
    scan_file,
    scan_paths,
)
from check_ownership import OwnershipConfig  # noqa: E402


# Helpers ----------------------------------------------------------------

def _write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _cfg() -> OwnershipConfig:
    """Use a small synthetic ownership table that mirrors the real one's
    structure but doesn't depend on its contents."""
    return OwnershipConfig(
        rules=(
            ("src/robolineage_contracts/", "contracts"),
            ("src/robolineage_schemas/", "contracts"),
            ("src/robolineage_data_source/", "data-source"),
            ("src/robolineage_shared_agents/visual_snapshot/", "visual-snapshot"),
            ("src/robolineage_train/", "training"),
            ("src/robolineage_dataset/", "dataset"),
        ),
        shared=(),
    )


# Same-plan import is OK -------------------------------------------------

def test_intra_plan_import_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(check_contracts_imports, "REPO_ROOT", tmp_path)
    f = _write(tmp_path, "src/robolineage_data_source/foo.py",
               "from robolineage_data_source.rosbag.recorder import RosbagRawRecorder\n")
    assert scan_file(f, _cfg()) == []


# robolineage_contracts is always allowed ----------------------------------------

@pytest.mark.parametrize("import_line", [
    "from robolineage_contracts.core import RolloutRecord",
    "from robolineage_contracts.agents import SnapshotAssessment",
    "import robolineage_contracts",
    "from robolineage_contracts import CONTRACTS_VERSION",
    "from robolineage_schemas import validate",
])
def test_robolineage_contracts_always_allowed(tmp_path, monkeypatch, import_line):
    monkeypatch.setattr(check_contracts_imports, "REPO_ROOT", tmp_path)
    f = _write(tmp_path, "src/robolineage_data_source/foo.py", import_line + "\n")
    assert scan_file(f, _cfg()) == []


# Cross-plan direct import is forbidden ----------------------------------

def test_cross_plan_import_flagged(tmp_path, monkeypatch):
    """Data-source code importing VSA internals is a violation."""
    monkeypatch.setattr(check_contracts_imports, "REPO_ROOT", tmp_path)
    f = _write(
        tmp_path,
        "src/robolineage_data_source/bad.py",
        "from robolineage_shared_agents.visual_snapshot.agent import VisualSnapshotAgent\n",
    )
    violations = scan_file(f, _cfg())
    assert len(violations) == 1
    v = violations[0]
    assert v.importer_owner == "data-source"
    assert v.target_owner == "visual-snapshot"
    assert v.target_module.startswith("robolineage_shared_agents.visual_snapshot")


def test_multiple_cross_plan_violations(tmp_path, monkeypatch):
    monkeypatch.setattr(check_contracts_imports, "REPO_ROOT", tmp_path)
    f = _write(
        tmp_path,
        "src/robolineage_data_source/x.py",
        """from robolineage_shared_agents.visual_snapshot.types import VisualObservation
from robolineage_train.trainer import TrainingRunner
""",
    )
    violations = scan_file(f, _cfg())
    assert len(violations) == 2
    assert {v.target_owner for v in violations} == {"visual-snapshot", "training"}


# Whitelist - training lifecycle may freeze datasets ---------------------

def test_whitelist_allows_dataset_updater_from_training_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(check_contracts_imports, "REPO_ROOT", tmp_path)
    f = _write(
        tmp_path,
        "src/robolineage_train/lifecycle.py",
        "from robolineage_dataset import DatasetUpdater\n",
    )
    assert scan_file(f, _cfg()) == []


def test_whitelist_does_not_cover_dataset_imports_from_other_training_files(tmp_path, monkeypatch):
    monkeypatch.setattr(check_contracts_imports, "REPO_ROOT", tmp_path)
    f = _write(
        tmp_path,
        "src/robolineage_train/x.py",
        "from robolineage_dataset import DatasetUpdater\n",
    )
    violations = scan_file(f, _cfg())
    assert len(violations) == 1


# Branch-import dirs are skipped -----------------------------------------

def test_branch_import_dirs_skipped(tmp_path, monkeypatch):
    """Files under `*_branch_import/` are exempt - they're temporary read-only
    snapshots that haven't been re-routed yet."""
    monkeypatch.setattr(check_contracts_imports, "REPO_ROOT", tmp_path)
    _write(
        tmp_path,
        "src/robolineage_data_source/stream_codex_branch_import/old.py",
        "from robolineage_train.trainer import TrainingRunner\n",
    )
    violations = scan_paths([tmp_path / "src/robolineage_data_source"], _cfg())
    assert violations == []


# Full src/ scan against current repo (real test) ------------------------

def test_real_src_no_violations():
    """Run the check against the live src/ tree on RoboLineage/H-contracts. Should be clean
    (contracts cannot import from any sibling runtime domain)."""
    from check_contracts_imports import scan_paths as real_scan
    from check_ownership import load_ownership as real_load

    cfg = real_load()
    violations = real_scan([REPO_ROOT / "src" / "robolineage_contracts"], cfg)
    assert violations == [], f"robolineage_contracts has cross-domain imports: {violations}"
