from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_ADAPTER_SCHEMA_VERSION = "RoboLineage.dataset_adapter_plan.v1"


@dataclass(frozen=True)
class DatasetAdapterPlan:
    """A data conversion plan chosen by RoboLineage for an external training repo."""

    adapter_id: str
    strategy: str
    confidence: float
    dataset_command: tuple[str, ...] | None = None
    train_input: str = "{dataset_output}"
    output_path: str = "{dataset_output}"
    source_data_policy: str = "read_only"
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = DATASET_ADAPTER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.dataset_command is not None:
            data["dataset_command"] = list(self.dataset_command)
        return data


class DatasetAdapterAgent:
    """Choose a dataset conversion strategy for an external training repo.

    The agent never mutates RoboLineage raw rollouts. It plans a converter that reads
    ``selected_rollouts.json`` and writes a new dataset under
    ``{dataset_output}``, leaving raw artifacts as source-of-truth evidence.
    """

    def plan(
        self,
        *,
        repo_root: Path,
        framework_type: str | None = None,
        dataset_command: str | tuple[str, ...] | None = None,
        train_command: str | tuple[str, ...] | None = None,
    ) -> DatasetAdapterPlan:
        repo = Path(repo_root).expanduser().resolve()
        if dataset_command is not None:
            return DatasetAdapterPlan(
                adapter_id="user_supplied_dataset_command",
                strategy="user_supplied",
                confidence=1.0,
                dataset_command=tuple(_command_parts(dataset_command)),
                assumptions=(
                    "User-provided dataset command is authoritative.",
                    "RoboLineage passes raw rollout references through selected_rollouts.json and does not mutate raw data.",
                ),
            )

        train_parts = _command_parts(train_command)
        if _command_uses(train_parts, "selected_rollouts_file"):
            return DatasetAdapterPlan(
                adapter_id="direct_selected_rollouts_file",
                strategy="direct_manifest",
                confidence=0.8,
                dataset_command=None,
                train_input="{selected_rollouts_file}",
                output_path="{selected_rollouts_file}",
                assumptions=("Training command consumes RoboLineage selected_rollouts.json directly.",),
            )

        if _command_uses(train_parts, "dataset_output"):
            return DatasetAdapterPlan(
                adapter_id="missing_dataset_converter",
                strategy="requires_dataset_command",
                confidence=0.2,
                dataset_command=None,
                warnings=(
                    "Training command references {dataset_output}, but no dataset conversion strategy was inferred.",
                    "Provide a dataset command or enable a repository-specific adapter.",
                ),
            )

        return DatasetAdapterPlan(
            adapter_id="no_conversion_required",
            strategy="no_conversion_required",
            confidence=0.5,
            dataset_command=None,
            assumptions=("Training command does not require RoboLineage to materialize {dataset_output}.",),
        )

    def write_plan(self, path: Path, plan: DatasetAdapterPlan) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _command_parts(command: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if command is None:
        return ()
    if isinstance(command, str):
        return tuple(shlex.split(command))
    return tuple(str(part) for part in command)


def _command_uses(parts: tuple[str, ...], placeholder: str) -> bool:
    token = "{" + placeholder + "}"
    return any(token in part for part in parts)
