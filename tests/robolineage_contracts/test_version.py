"""H0 — version constant smoke."""
import re

from robolineage_contracts import CONTRACTS_VERSION
from robolineage_contracts.version import CONTRACTS_VERSION as V_INNER


def test_version_is_semver():
    assert re.match(r"^\d+\.\d+\.\d+$", CONTRACTS_VERSION), (
        f"CONTRACTS_VERSION must be SemVer; got {CONTRACTS_VERSION!r}"
    )


def test_top_level_reexports_match_inner():
    assert CONTRACTS_VERSION == V_INNER


def test_version_is_v0_4_0():
    """Sanity: while v0.4.0 (compatibility-only contract removal) is current, lock
    to it. Bumping this test is intentional + signals a deliberate version
    change."""
    assert CONTRACTS_VERSION == "0.4.0"
