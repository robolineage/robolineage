"""ValidationIssue — unified issue shape used everywhere validation runs.

Source of truth: doc/agent contract.md §common conventions.

Used by:
- dataset/training validation — file/format checks on rollout dirs
- post-review / legacy metadata validation findings
- robolineage_schemas.validate() — JSON Schema violations
- validate_metadata_transition() (in core.metadata) — write-stage permission

The `code` field is a snake_case identifier (e.g. "missing_file",
"pose_width_mismatch"). For schema violations the convention is
"schema:<keyword>" (e.g. "schema:required"). The regex is enforced in
`__post_init__` to catch drift before it reaches consumers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


Severity = Literal["error", "warning"]

# snake_case identifier, optionally followed by ":<sub-id>" (used for
# "schema:required", "schema:type", etc.). Sub-id may include hyphens.
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*(:[a-z][a-z0-9_-]*)?$")


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding.

    Attributes:
        severity: "error" (blocking) or "warning" (advisory).
        code:     Stable machine-readable identifier. snake_case, optionally
                  prefixed with "schema:" for JSON-Schema-derived issues.
        message:  Human-readable detail. Include path/value context when
                  helpful, but avoid putting it in `code` so callers can
                  group by `code`.
    """
    severity: Severity
    code: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in ("error", "warning"):
            raise ValueError(
                f"severity must be 'error' or 'warning'; got {self.severity!r}"
            )
        if not _CODE_RE.match(self.code):
            raise ValueError(
                "code must be snake_case (or 'schema:<keyword>'); "
                f"got {self.code!r}"
            )
