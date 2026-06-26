"""Bidirectional Pydantic ↔ JSON-ready dict helpers.

Used when:
  - reading JSON-from-disk and want a typed model: `to_model(d, MetadataModel)`
  - dumping a typed model to JSON: `to_jsonable(model)`

These are thin wrappers around `model_validate` / `model_dump`. They exist
mostly so the import path is stable (callers never need to remember whether
to call `parse_obj` vs `model_validate` etc. across Pydantic versions).
"""
from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def to_model(instance: dict, model_class: Type[T]) -> T:
    """Parse a JSON-ready dict into a Pydantic model.

    Raises pydantic.ValidationError on type / constraint violations.
    """
    return model_class.model_validate(instance)


def to_jsonable(model_instance: BaseModel) -> dict:
    """Dump a Pydantic model to a JSON-ready dict (alias-free, includes None)."""
    return model_instance.model_dump(exclude_none=False)
