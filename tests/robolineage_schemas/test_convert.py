"""H1 — to_model / to_jsonable round-trip."""
from __future__ import annotations

import json
from pathlib import Path

from robolineage_contracts.core import MetadataModel
from robolineage_schemas.convert import to_model, to_jsonable


FIXTURES = Path(__file__).resolve().parents[1] / "_shared_fixtures"


def test_round_trip_preserves_required_fields():
    raw = json.loads((FIXTURES / "mini_rollout" / "metadata.json").read_text())
    m = to_model(raw, MetadataModel)
    dumped = to_jsonable(m)

    # Required top-level keys preserved
    for key in ("exportId", "project", "task", "annotation", "dataPackage",
                "exportedAt", "exportedBy", "alignment"):
        assert key in dumped, f"round-trip dropped {key}"

    # Critical nested fields preserved
    assert dumped["annotation"]["review"]["score"] == raw["annotation"]["review"]["score"]
    assert dumped["alignment"]["referenceCam"] == raw["alignment"]["referenceCam"]


def test_to_jsonable_emits_none_for_optional_l1():
    raw = json.loads((FIXTURES / "mini_rollout" / "metadata.json").read_text())
    m = to_model(raw, MetadataModel)
    dumped = to_jsonable(m)
    # Fixture has annotation.l1 = null; round-trip preserves explicit None.
    assert dumped["annotation"]["l1"] is None
