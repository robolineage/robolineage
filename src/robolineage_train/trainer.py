from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from robolineage_contracts.pipeline import DatasetLock


@dataclass(frozen=True)
class TrainingConfig:
    policy_version: str
    architecture: str
    dataset_lock_path: Path
    output_dir: Path
    command: list[str]
    training_steps: int | None = None


@dataclass(frozen=True)
class TrainingResult:
    policy_bin_path: Path
    log_path: Path
    steps: int


class Trainer:
    """Wrapper around an external policy-training command."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def run(self) -> TrainingResult:
        _read_dataset_lock(self.config.dataset_lock_path)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            part.format(
                output_dir=str(output_dir),
                dataset_lock=str(self.config.dataset_lock_path),
                policy_version=self.config.policy_version,
            )
            for part in self.config.command
        ]
        env = os.environ.copy()
        env.update(
            {
                "ROBOLINEAGE_POLICY_OUTPUT_DIR": str(output_dir),
                "ROBOLINEAGE_DATASET_LOCK": str(self.config.dataset_lock_path),
                "ROBOLINEAGE_POLICY_VERSION": self.config.policy_version,
            }
        )
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        policy_bin = output_dir / "policy.bin"
        training_log = output_dir / "training_log.txt"
        if not policy_bin.exists():
            raise FileNotFoundError(f"training command did not write {policy_bin}")
        if not training_log.exists():
            raise FileNotFoundError(f"training command did not write {training_log}")

        steps = self.config.training_steps
        if steps is None:
            steps = _parse_steps(result.stdout + "\n" + result.stderr)
        return TrainingResult(policy_bin_path=policy_bin, log_path=training_log, steps=steps)


def _parse_steps(text: str) -> int:
    match = re.search(r"(?:training_steps|steps)\s*=\s*(\d+)", text)
    return int(match.group(1)) if match else 0


def _read_dataset_lock(path: Path) -> DatasetLock:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["included_rollout_ids"] = tuple(raw["included_rollout_ids"])
    return DatasetLock(**raw)
