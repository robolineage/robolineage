from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from robolineage_contracts.agents import ValidationIssue
from robolineage_contracts.pipeline import DatasetLock, PolicyMeta

from .trainer import TrainingConfig, TrainingResult


class PolicyMetaWriter:
    def write(
        self,
        *,
        trainer_result: TrainingResult,
        dataset_lock: DatasetLock,
        training_config: TrainingConfig,
        eval_success_rate: float | None = None,
        gating_result: Literal["pass", "fail", "pending"] = "pending",
    ) -> Path:
        if dataset_lock.version_id == "":
            raise ValueError("dataset_lock.version_id must be non-empty")

        meta = PolicyMeta(
            version_id=training_config.policy_version,
            trained_on_dataset=dataset_lock.version_id,
            architecture=training_config.architecture,
            training_steps=trainer_result.steps,
            created_at=datetime.now(timezone.utc).isoformat(),
            eval_success_rate=eval_success_rate,
            deployed=False,
            deployment_gating_result=gating_result,
        )
        target = Path(training_config.output_dir) / "policy.meta.json"
        if target.exists():
            raise FileExistsError(f"policy.meta.json already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name("policy.meta.json.tmp")
        tmp.write_text(
            json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(target)
        target.chmod(0o444)
        return target

    def write_framework_meta(
        self,
        *,
        policy_version: str,
        architecture: str,
        dataset_lock: DatasetLock,
        checkpoint_dir: Path,
        checkpoint_path: Path | None,
        training_steps: int,
        eval_success_rate: float | None,
        framework_name: str,
        framework_type: str,
        adapter_version: str,
        training_result_path: Path,
        eval_result_path: Path | None = None,
        ROBOLINEAGE_context_path: Path | None = None,
        gating_result: Literal["pass", "fail", "pending"] = "pending",
    ) -> Path:
        meta = PolicyMeta(
            version_id=policy_version,
            trained_on_dataset=dataset_lock.version_id,
            architecture=architecture,
            training_steps=training_steps,
            created_at=datetime.now(timezone.utc).isoformat(),
            eval_success_rate=eval_success_rate,
            deployed=False,
            deployment_gating_result=gating_result,
            framework_name=framework_name,
            framework_type=framework_type,
            adapter_version=adapter_version,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            training_result_path=str(training_result_path),
            eval_result_path=str(eval_result_path) if eval_result_path is not None else None,
            ROBOLINEAGE_context_path=str(ROBOLINEAGE_context_path) if ROBOLINEAGE_context_path is not None else None,
        )
        target = Path(checkpoint_dir) / "policy.meta.json"
        if target.exists():
            raise FileExistsError(f"policy.meta.json already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name("policy.meta.json.tmp")
        tmp.write_text(
            json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(target)
        target.chmod(0o444)
        return target

    @staticmethod
    def verify_integrity(
        meta_path: Path,
        dataset_lock: DatasetLock,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [
                ValidationIssue(
                    severity="error",
                    code="policy_meta_unreadable",
                    message=f"cannot read policy meta: {exc}",
                )
            ]

        trained_on_dataset = meta.get("trained_on_dataset")
        if trained_on_dataset != dataset_lock.version_id:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="policy_source_dataset_mismatch",
                    message=(
                        f"meta.trained_on_dataset={trained_on_dataset!r} "
                        f"!= dataset_lock.version_id={dataset_lock.version_id!r}"
                    ),
                )
            )
        return issues
