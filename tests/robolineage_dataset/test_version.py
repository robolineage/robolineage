from __future__ import annotations

import pytest

from robolineage_dataset.version import next_version_id, parse_version


def test_parse_version():
    assert parse_version("v3") == 3


def test_next_version_id():
    assert next_version_id(None) == "v1"
    assert next_version_id("v1") == "v2"


def test_invalid_version_rejected():
    with pytest.raises(ValueError, match="Invalid version"):
        parse_version("version3")
