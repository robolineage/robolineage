from __future__ import annotations

import re


_VERSION_RE = re.compile(r"^v(\d+)$")


def parse_version(version_id: str) -> int:
    match = _VERSION_RE.match(version_id)
    if match is None:
        raise ValueError(f"Invalid version id: {version_id!r}")
    return int(match.group(1))


def next_version_id(prev: str | None) -> str:
    if prev is None:
        return "v1"
    return f"v{parse_version(prev) + 1}"
