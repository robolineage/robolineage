"""RoboLineage type contracts — single source of truth for cross-domain dataclasses,
enums, Pydantic models, and JSON schemas.

This package is **declarative only**: no business logic, no hardware deps,
no LLM calls. It owns the Python form of every contract documented in `doc/`.

Import shape:
    from robolineage_contracts import CONTRACTS_VERSION
    from robolineage_contracts.core import RolloutRecord, MetadataModel
    from robolineage_contracts.agents import SnapshotAssessment, ValidationIssue

Sub-packages:
    core    — RolloutRecord / MetadataModel + transition validator
    agents  — SnapshotAssessment / ValidationIssue
    pipeline — DatasetLock / PolicyMeta / manifest entries
    session — SessionState / ControlEvent / FeedbackEvent

Cross-domain rule: shared type usage must go through `robolineage_contracts.*` or
`robolineage_schemas.*`. Runtime composition exceptions are explicit in
`scripts/check_contracts_imports.py`.
"""
from robolineage_contracts.version import CONTRACTS_VERSION

__all__ = ["CONTRACTS_VERSION"]
