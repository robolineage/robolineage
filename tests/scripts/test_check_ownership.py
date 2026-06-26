"""H6 - check_ownership.py: ownership lookup, --expected-owner, shared exemption."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_ownership  # noqa: E402
from check_ownership import (  # noqa: E402
    OwnershipConfig,
    check,
    find_owner,
    load_ownership,
    main,
    _matches_shared,
    _parse_yaml_minimal,
)


# -- _parse_yaml_minimal --------------------------------------------------

def test_parse_yaml_basic():
    text = """
rules:
  - path: src/robolineage_contracts/
    owner: H
  - path: src/robolineage_data_source/
    owner: A

shared:
  - AGENTS.md
  - pyproject.toml
"""
    cfg = _parse_yaml_minimal(text)
    assert cfg.rules == (
        ("src/robolineage_contracts/", "H"),
        ("src/robolineage_data_source/", "A"),
    )
    assert "AGENTS.md" in cfg.shared
    assert "pyproject.toml" in cfg.shared


def test_parse_yaml_strips_comments():
    text = """
rules:
  - path: src/robolineage_contracts/    # main contracts
    owner: H
shared:
  - AGENTS.md   # cross-domain metadata
"""
    cfg = _parse_yaml_minimal(text)
    assert ("src/robolineage_contracts/", "H") in cfg.rules
    assert "AGENTS.md" in cfg.shared


def test_parse_yaml_ignores_blank_lines():
    text = """

rules:

  - path: x/
    owner: H

shared:

  - y
"""
    cfg = _parse_yaml_minimal(text)
    assert cfg.rules == (("x/", "H"),)


def test_parse_yaml_real_file():
    cfg = load_ownership()
    assert any(p == "src/robolineage_contracts/" and o == "H" for p, o in cfg.rules)
    assert "AGENTS.md" in cfg.shared


def test_parse_yaml_rejects_conflict(tmp_path):
    bad = tmp_path / "OWNERSHIP.yaml"
    bad.write_text("""
rules:
  - path: src/x/
    owner: A
  - path: src/x/
    owner: B
""", encoding="utf-8")
    with pytest.raises(SystemExit, match="conflict"):
        load_ownership(bad)


# -- find_owner: longest-prefix matching ----------------------------------

def _cfg(*rules: tuple[str, str]) -> OwnershipConfig:
    return OwnershipConfig(rules=tuple(rules), shared=())


def test_find_owner_simple():
    cfg = _cfg(("src/robolineage_contracts/", "H"))
    assert find_owner("src/robolineage_contracts/core/rollout.py", cfg) == "H"


def test_find_owner_longest_prefix_wins():
    """A more-specific prefix should override a less-specific one."""
    cfg = _cfg(
        ("src/robolineage_shared_agents/", "OTHER"),  # broad
        ("src/robolineage_shared_agents/visual_snapshot/", "B"),
    )
    assert find_owner("src/robolineage_shared_agents/visual_snapshot/agent.py", cfg) == "B"


def test_find_owner_returns_none_when_unmatched():
    cfg = _cfg(("src/robolineage_contracts/", "H"))
    assert find_owner("src/some_other/x.py", cfg) is None


def test_find_owner_exact_file_path():
    """Exact-path rules (not just dir prefixes) work."""
    cfg = _cfg(("src/robolineage_shared_agents/__init__.py", "H"))
    assert find_owner("src/robolineage_shared_agents/__init__.py", cfg) == "H"


# -- _matches_shared ------------------------------------------------------

def test_shared_exact_match():
    assert _matches_shared("AGENTS.md", ("AGENTS.md",))


def test_shared_dir_prefix():
    assert _matches_shared("doc/foo.md", ("doc/",))
    assert _matches_shared("doc/sub/bar.md", ("doc/",))


def test_shared_glob_with_star():
    assert _matches_shared("scripts/foo.sh", ("scripts/*.sh",))
    assert not _matches_shared("scripts/foo.py", ("scripts/*.sh",))


def test_shared_no_match():
    assert not _matches_shared("AGENTS.md", ("AGENTS.md",))


# -- check() --------------------------------------------------------------

@pytest.fixture
def small_cfg() -> OwnershipConfig:
    return OwnershipConfig(
        rules=(
            ("src/robolineage_contracts/", "H"),
            ("src/robolineage_data_source/", "A"),
            ("src/robolineage_shared_agents/__init__.py", "H"),
            ("src/robolineage_shared_agents/visual_snapshot/", "B"),
        ),
        shared=("AGENTS.md", "pyproject.toml"),
    )


def test_check_ok(small_cfg):
    findings = check(["src/robolineage_contracts/version.py"], small_cfg, expected_owner="H")
    assert all(f.kind == "ok" for f in findings)


def test_check_unowned_file(small_cfg):
    findings = check(["src/random/path.py"], small_cfg)
    assert findings[0].kind == "unowned"


def test_check_wrong_owner(small_cfg):
    findings = check(
        ["src/robolineage_data_source/x.py"],
        small_cfg,
        expected_owner="H",  # but the file belongs to A
    )
    assert findings[0].kind == "wrong_owner"
    assert findings[0].owner == "A"


def test_check_shared_exempt(small_cfg):
    findings = check(["AGENTS.md"], small_cfg, expected_owner="H")
    assert findings[0].kind == "shared"


def test_check_mixed(small_cfg):
    findings = check(
        [
            "src/robolineage_contracts/version.py",   # ok
            "src/robolineage_data_source/x.py",       # wrong (when expected=H)
            "AGENTS.md",                       # shared
            "totally/unowned.py",             # unowned
        ],
        small_cfg,
        expected_owner="H",
    )
    kinds = {f.path: f.kind for f in findings}
    assert kinds == {
        "src/robolineage_contracts/version.py": "ok",
        "src/robolineage_data_source/x.py": "wrong_owner",
        "AGENTS.md": "shared",
        "totally/unowned.py": "unowned",
    }


# -- main() exit codes ----------------------------------------------------

def test_main_exit_code_zero(capsys):
    rc = main(["--files", "src/robolineage_contracts/version.py", "--expected-owner", "H"])
    assert rc == 0


def test_main_exit_code_one_when_unowned(capsys):
    rc = main(["--files", "totally/unowned.py"])
    assert rc == 1


def test_main_exit_code_two_when_wrong_owner(capsys):
    rc = main([
        "--files", "src/robolineage_data_source/x.py",
        "--expected-owner", "H",
    ])
    assert rc == 2


def test_main_explicit_files_with_shared(capsys):
    """Shared files don't trigger wrong_owner."""
    rc = main([
        "--files", "src/robolineage_contracts/version.py", "AGENTS.md", "pyproject.toml",
        "--expected-owner", "H",
    ])
    assert rc == 0
