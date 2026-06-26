"""JSON Schema validation entry point.

Loads a schema from this package by stem name and runs Draft 2020-12
validation. All findings are normalised to `ValidationIssue` so callers
can mix schema errors and contract errors in one list.

Public:
    load_schema(name) -> dict        # name is e.g. "metadata", "snapshot"
    validate(instance, schema_name) -> list[ValidationIssue]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from jsonschema import Draft202012Validator

from robolineage_contracts.agents.validation import ValidationIssue


_SCHEMAS_DIR = Path(__file__).parent


def load_schema(name: str) -> dict:
    """Load a JSON schema by stem name.

    Looks up `<package_dir>/<name>.schema.json`. Raises FileNotFoundError if
    the schema does not exist (intentional — typo'd name should fail loudly).
    """
    path = _SCHEMAS_DIR / f"{name}.schema.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Schema not found: {path}. Available: "
            f"{sorted(p.stem.removesuffix('.schema') for p in _SCHEMAS_DIR.glob('*.schema.json'))}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def validate(instance: Any, schema_name: str) -> List[ValidationIssue]:
    """Validate `instance` against the named schema.

    Returns a (possibly empty) list of ValidationIssue. Empty list means
    the instance is conformant. Issues from schema validation are tagged
    with `code = "schema:<keyword>"` (e.g. "schema:required", "schema:enum").
    """
    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema)
    issues: list[ValidationIssue] = []
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        path_str = "/".join(str(p) for p in err.absolute_path) or "<root>"
        issues.append(
            ValidationIssue(
                severity="error",
                code=f"schema:{err.validator}",
                message=f"{path_str}: {err.message}",
            )
        )
    return issues
