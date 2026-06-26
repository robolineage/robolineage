"""Rollout id helpers for RoboLineage sessions."""
from __future__ import annotations

import re
import uuid


_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def generate_rollout_id() -> str:
    return str(uuid.uuid4())


def is_valid_rollout_id(value: str) -> bool:
    return bool(_UUID4_RE.match(value))
