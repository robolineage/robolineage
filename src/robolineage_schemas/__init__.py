"""JSON Schema files + validation utilities for `robolineage_contracts`.

Each `<name>.schema.json` here mirrors a contract that also has a Python
class in `robolineage_contracts.*`. Use them when you need wire-format validation
(JSON-on-disk, JSONL streams, HTTP payloads) — for in-memory typing prefer
the Pydantic models / dataclasses in `robolineage_contracts.*`.

Public API:

    from robolineage_schemas import validate, load_schema
    from robolineage_schemas.artifacts import write_validated_json_atomic
    from robolineage_schemas.convert import to_model, to_jsonable

    issues = validate(some_dict, "metadata")
    if any(i.severity == "error" for i in issues):
        raise SomeError(issues)

`validate(...)` returns `list[robolineage_contracts.agents.ValidationIssue]`.

See `src/robolineage_schemas/README.md` for the file inventory and which Plan owns
each schema.
"""
from robolineage_schemas.artifacts import (
    ArtifactValidationError,
    validate_artifact_payload,
    write_validated_json_atomic,
)
from robolineage_schemas.validate import validate, load_schema

__all__ = [
    "validate",
    "load_schema",
    "ArtifactValidationError",
    "validate_artifact_payload",
    "write_validated_json_atomic",
]
