from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robolineage_contracts.agents.validation import ValidationIssue
from robolineage_schemas.validate import validate


class ArtifactValidationError(ValueError):
    """Raised when an on-disk artifact payload fails its JSON schema."""

    def __init__(self, schema_name: str, issues: list[ValidationIssue]) -> None:
        self.schema_name = schema_name
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        super().__init__(f"{schema_name} artifact failed schema validation: {details}")


def validate_artifact_payload(payload: Any, schema_name: str) -> None:
    errors = [issue for issue in validate(payload, schema_name) if issue.severity == "error"]
    if errors:
        raise ArtifactValidationError(schema_name, errors)


def write_validated_json_atomic(path: str | Path, payload: dict[str, Any], schema_name: str) -> Path:
    """Validate a JSON artifact before atomically replacing it on disk."""

    validate_artifact_payload(payload, schema_name)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target
