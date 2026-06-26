from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robolineage_contracts.pipeline import DatasetLock, TrainManifestEntry
from robolineage_dataset import DatasetUpdater
from robolineage_dataset.version import parse_version

from .framework_adapter import FrameworkAdapter, FrameworkProfile, FrameworkRunResult
from .dataset_health import DatasetHealthAgent
from .policy_meta import PolicyMetaWriter


LIFECYCLE_SCHEMA_VERSION = "RoboLineage.training_lifecycle.v1"
RECOMMENDATION_SCHEMA_VERSION = "RoboLineage.deployment_recommendation.v1"


@dataclass(frozen=True)
class TrainingLifecycleResult:
    run_id: str
    workspace_dir: Path
    dataset_lock_path: Path
    dataset_version: str
    policy_meta_path: Path
    ROBOLINEAGE_context_path: Path
    deployment_recommendation_path: Path
    training_result_path: Path


@dataclass(frozen=True)
class TrainingDatasetAdaptResult:
    run_id: str
    workspace_dir: Path
    dataset_lock_path: Path
    dataset_version: str
    dataset_adapt_status_path: Path
    dataset_adapt_result_path: Path
    dataset_health_report_path: Path | None = None
    dataset_health_understanding_path: Path | None = None


class TrainingLifecycleRunner:
    """Post-review → dataset version → framework run → policy lifecycle."""

    def __init__(
        self,
        *,
        profile: FrameworkProfile,
        rollouts_root: Path,
        datasets_root: Path,
        workspace_root: Path,
        prev_lock_path: Path | None = None,
        include_decisions: tuple[str, ...] = ("accepted",),
        include_rollout_ids: tuple[str, ...] | None = None,
        deploy_success_threshold: float = 0.7,
        dataset_health_llm_client: Any | None = None,
        enable_dataset_health_llm: bool | None = None,
    ) -> None:
        self.profile = profile
        self.rollouts_root = Path(rollouts_root)
        self.datasets_root = Path(datasets_root)
        self.workspace_root = Path(workspace_root)
        self.prev_lock_path = prev_lock_path
        self.include_decisions = include_decisions
        self.include_rollout_ids = include_rollout_ids
        self.deploy_success_threshold = deploy_success_threshold
        self._dataset_health_llm_client = dataset_health_llm_client
        self._enable_dataset_health_llm = True if enable_dataset_health_llm is None else bool(enable_dataset_health_llm)
        self._enable_dataset_health_env_llm = (
            bool(os.environ.get("DATASET_HEALTH_LLM_API_KEY"))
            if enable_dataset_health_llm is None
            else bool(enable_dataset_health_llm)
        )

    def run(
        self,
        *,
        policy_version: str,
        architecture: str | None = None,
        run_id: str | None = None,
    ) -> TrainingLifecycleResult:
        adapted = self.adapt_data(policy_version=policy_version, run_id=run_id)
        return self.train_adapted(
            policy_version=policy_version,
            architecture=architecture,
            run_id=adapted.run_id,
            dataset_version=adapted.dataset_version,
            dataset_lock_path=adapted.dataset_lock_path,
        )

    def adapt_data(
        self,
        *,
        policy_version: str,
        run_id: str | None = None,
    ) -> TrainingDatasetAdaptResult:
        run_id = run_id or _run_id(self.profile.name, policy_version)
        workspace = self.workspace_root / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        manifest_path = workspace / "train_manifest.jsonl"
        entries = build_train_manifest_from_post_review(
            rollouts_root=self.rollouts_root,
            output_path=manifest_path,
            include_decisions=self.include_decisions,
            include_rollout_ids=self.include_rollout_ids,
        )
        health_inputs, health_task_config = _dataset_health_inputs(entries)
        health_result = DatasetHealthAgent(
            llm_client=self._dataset_health_llm_client,
            enable_llm_understanding=self._enable_dataset_health_llm,
            enable_env_llm=self._enable_dataset_health_env_llm,
        ).analyze(
            selected_rollouts=health_inputs,
            dataset_history={},
            task_config=health_task_config,
            output_dir=workspace,
        )
        prev_lock_path = self.prev_lock_path or latest_dataset_lock(self.datasets_root)
        lock = DatasetUpdater().update(
            train_manifest_path=manifest_path,
            prev_lock_path=prev_lock_path,
            out_dir=self.datasets_root,
            changelog=f"RoboLineage framework run {run_id}: {len(entries)} selected rollouts",
        )
        dataset_lock_path = self.datasets_root / lock.version_id / "dataset.lock"

        adapt_result = FrameworkAdapter(self.profile).adapt_dataset(
            rollouts_root=self.rollouts_root,
            workspace_dir=workspace / "framework",
            dataset_version=lock.version_id,
            policy_version=policy_version,
            include_decisions=self.include_decisions,
            include_rollout_ids=self.include_rollout_ids,
            symlink_selected=None,
        )
        return TrainingDatasetAdaptResult(
            run_id=run_id,
            workspace_dir=workspace,
            dataset_lock_path=dataset_lock_path,
            dataset_version=lock.version_id,
            dataset_adapt_status_path=adapt_result.dataset_adapt_status_path,
            dataset_adapt_result_path=adapt_result.dataset_adapt_result_path,
            dataset_health_report_path=health_result.report_path,
            dataset_health_understanding_path=health_result.understanding_path,
        )

    def train_adapted(
        self,
        *,
        policy_version: str,
        architecture: str | None = None,
        run_id: str,
        dataset_version: str,
        dataset_lock_path: Path,
    ) -> TrainingLifecycleResult:
        workspace = self.workspace_root / run_id
        lock = read_dataset_lock(dataset_lock_path)
        framework_result = FrameworkAdapter(self.profile).run_training_only(
            workspace_dir=workspace / "framework",
            dataset_version=dataset_version,
            policy_version=policy_version,
        )
        return self._finalize_training(
            run_id=run_id,
            workspace=workspace,
            dataset_lock_path=dataset_lock_path,
            lock=lock,
            framework_result=framework_result,
            policy_version=policy_version,
            architecture=architecture,
        )

    def _finalize_training(
        self,
        *,
        run_id: str,
        workspace: Path,
        dataset_lock_path: Path,
        lock: DatasetLock,
        framework_result: FrameworkRunResult,
        policy_version: str,
        architecture: str | None,
    ) -> TrainingLifecycleResult:
        eval_payload = _read_json(framework_result.eval_result_path)
        eval_success_rate = _success_rate(eval_payload)
        manifest_path = workspace / "train_manifest.jsonl"
        selected_manifest_entries = len(_read_jsonl(manifest_path)) if manifest_path.exists() else 0
        recommendation = deployment_recommendation(
            framework_result_path=framework_result.training_result_path,
            eval_payload=eval_payload,
            success_threshold=self.deploy_success_threshold,
        )
        recommendation_path = workspace / "deployment_recommendation.json"
        _write_json_atomic(recommendation_path, recommendation)

        ROBOLINEAGE_context_path = workspace / "policy.ROBOLINEAGE_context.json"
        context = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "run_id": run_id,
            "framework": {
                "name": self.profile.name,
                "framework_type": self.profile.framework_type,
                "adapter_version": self.profile.adapter_version,
                "repo_root": str(self.profile.repo_root),
                "staging": asdict(self.profile.staging),
                "dataset_adapter": self.profile.dataset_adapter,
            },
            "dataset": {
                "version_id": lock.version_id,
                "dataset_lock_path": str(dataset_lock_path),
                "manifest_path": str(manifest_path),
                "selected_manifest_entries": selected_manifest_entries,
                "selection_rollout_ids": list(self.include_rollout_ids or []),
                "dataset_health_report_path": str(workspace / "dataset_health_report.json"),
                "dataset_health_understanding_path": str(workspace / "dataset_health_understanding.json"),
            },
            "post_review_summary": post_review_summary(
                self.rollouts_root,
                self.include_decisions,
                include_rollout_ids=self.include_rollout_ids,
            ),
            "framework_run": _read_json(framework_result.training_result_path),
            "training_status": _read_json(framework_result.training_status_path),
            "eval_result": eval_payload,
            "deployment_recommendation": recommendation,
            "created_at": _now_iso(),
        }
        _write_json_atomic(ROBOLINEAGE_context_path, context)

        training_status = _read_json(framework_result.training_status_path) or {}
        metrics = training_status.get("metrics") if isinstance(training_status.get("metrics"), dict) else {}
        steps = _int_or_zero(metrics.get("latest_step") or metrics.get("step") or metrics.get("steps"))
        policy_meta_path = PolicyMetaWriter().write_framework_meta(
            policy_version=policy_version,
            architecture=architecture or self.profile.framework_type,
            dataset_lock=lock,
            checkpoint_dir=framework_result.checkpoint_dir,
            checkpoint_path=framework_result.checkpoint_path,
            training_steps=steps,
            eval_success_rate=eval_success_rate,
            framework_name=self.profile.name,
            framework_type=self.profile.framework_type,
            adapter_version=self.profile.adapter_version,
            training_result_path=framework_result.training_result_path,
            eval_result_path=framework_result.eval_result_path,
            ROBOLINEAGE_context_path=ROBOLINEAGE_context_path,
            gating_result=recommendation["gating_result"],
        )
        return TrainingLifecycleResult(
            run_id=run_id,
            workspace_dir=workspace,
            dataset_lock_path=dataset_lock_path,
            dataset_version=lock.version_id,
            policy_meta_path=policy_meta_path,
            ROBOLINEAGE_context_path=ROBOLINEAGE_context_path,
            deployment_recommendation_path=recommendation_path,
            training_result_path=framework_result.training_result_path,
        )


def build_train_manifest_from_post_review(
    *,
    rollouts_root: Path,
    output_path: Path,
    include_decisions: tuple[str, ...] = ("accepted",),
    include_rollout_ids: tuple[str, ...] | None = None,
) -> list[TrainManifestEntry]:
    allowed = set(include_decisions)
    allowed_rollouts = set(include_rollout_ids) if include_rollout_ids is not None else None
    entries: list[TrainManifestEntry] = []
    for rollout_dir in sorted(Path(rollouts_root).iterdir()):
        if not rollout_dir.is_dir():
            continue
        if allowed_rollouts is not None and rollout_dir.name not in allowed_rollouts:
            continue
        admission = _read_json(rollout_dir / "dataset_admission.json")
        if not admission:
            continue
        accepted_for_training = admission.get("accepted_for_training")
        if isinstance(accepted_for_training, bool):
            if not accepted_for_training:
                continue
        elif str(admission.get("decision") or "") not in allowed:
            continue
        summary = _read_json(rollout_dir / "rollout_summary.json") or {}
        annotation = _read_json(rollout_dir / "annotation.final.json") or {}
        reasons = tuple(str(item) for item in admission.get("reasons") or ())
        phases = _annotation_phases(annotation)
        confidence = _float_or_default(summary.get("success_confidence"), 0.5)
        entries.append(
            TrainManifestEntry(
                export_id=rollout_dir.name,
                rollout_id=rollout_dir.name,
                sample_dir=str(rollout_dir.resolve()),
                review_score=_review_score_for_decision(str(admission.get("decision") or "")),
                confidence=confidence,
                l1_phases=phases,
                reasons=reasons,
            )
        )
    _write_jsonl_atomic(output_path, [asdict(entry) for entry in entries])
    return entries


def post_review_summary(
    rollouts_root: Path,
    include_decisions: tuple[str, ...],
    *,
    include_rollout_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    allowed = set(include_decisions)
    allowed_rollouts = set(include_rollout_ids) if include_rollout_ids is not None else None
    selected = 0
    failure_types: dict[str, int] = {}
    data_use: dict[str, int] = {}
    for rollout_dir in sorted(Path(rollouts_root).iterdir()):
        if not rollout_dir.is_dir():
            continue
        if allowed_rollouts is not None and rollout_dir.name not in allowed_rollouts:
            continue
        admission = _read_json(rollout_dir / "dataset_admission.json")
        if not admission:
            continue
        accepted_for_training = admission.get("accepted_for_training")
        if isinstance(accepted_for_training, bool):
            if not accepted_for_training:
                continue
        elif str(admission.get("decision") or "") not in allowed:
            continue
        selected += 1
        for item in admission.get("data_use") or ():
            data_use[str(item)] = data_use.get(str(item), 0) + 1
        failure = _read_json(rollout_dir / "failure_analysis.json") or {}
        for event in failure.get("failure_events") or ():
            if isinstance(event, dict):
                label = str(event.get("failure_type") or "unknown")
                failure_types[label] = failure_types.get(label, 0) + 1
    return {
        "selected_rollout_count": selected,
        "include_decisions": list(include_decisions),
        "include_rollout_ids": list(include_rollout_ids or []),
        "failure_type_counts": failure_types,
        "data_use_counts": data_use,
    }


def _dataset_health_inputs(entries: list[TrainManifestEntry]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    expected_phases: list[str] = []
    task_description = ""
    for entry in entries:
        rollout_dir = Path(entry.sample_dir)
        admission = _read_json(rollout_dir / "dataset_admission.json") or {}
        summary = _read_json(rollout_dir / "rollout_summary.json") or {}
        annotation = _read_json(rollout_dir / "annotation.final.json") or {}
        failure = _read_json(rollout_dir / "failure_analysis.json") or {}
        phases = tuple(entry.l1_phases or ()) or (_annotation_phases(annotation) or ())
        if phases and not expected_phases:
            expected_phases = [str(item) for item in phases]
        task = annotation.get("task") if isinstance(annotation.get("task"), dict) else {}
        if not expected_phases and isinstance(task.get("phases"), list):
            expected_phases = [str(item) for item in task.get("phases") or []]
        task_description = task_description or str(
            summary.get("task_description") or admission.get("task_description") or ""
        )
        selected.append(
            {
                "rollout_id": entry.rollout_id,
                "decision": admission.get("decision"),
                "data_use": admission.get("data_use") or [],
                "final_success": summary.get("final_success", summary.get("success_likely")),
                "task_description": task_description,
                "phases_seen": [str(item) for item in phases],
                "failure_analysis": failure,
            }
        )
    return selected, {"phases": expected_phases, "task_description": task_description}


def latest_dataset_lock(datasets_root: Path) -> Path | None:
    root = Path(datasets_root)
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        lock = path / "dataset.lock"
        if not lock.exists():
            continue
        try:
            version_num = parse_version(path.name)
        except ValueError:
            continue
        candidates.append((version_num, lock))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def read_dataset_lock(path: Path) -> DatasetLock:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["included_rollout_ids"] = tuple(raw.get("included_rollout_ids") or ())
    return DatasetLock(**raw)


def deployment_recommendation(
    *,
    framework_result_path: Path,
    eval_payload: dict[str, Any] | None,
    success_threshold: float = 0.7,
) -> dict[str, Any]:
    training_result = _read_json(framework_result_path) or {}
    commands = training_result.get("commands") if isinstance(training_result.get("commands"), list) else []
    train_reports = [
        cmd.get("parsed") for cmd in commands
        if isinstance(cmd, dict) and cmd.get("name") == "train" and isinstance(cmd.get("parsed"), dict)
    ]
    train_report = train_reports[-1] if train_reports else {}
    status = str(train_report.get("status") or "unknown")
    warnings = list(train_report.get("warnings") or [])
    errors = list(train_report.get("errors") or [])
    success_rate = _success_rate(eval_payload)
    reasons: list[str] = []
    decision = "hold"
    gating_result = "pending"

    if errors or status in {"failed", "unstable"}:
        decision = "hold"
        gating_result = "fail"
        reasons.append("training_monitor_reported_errors_or_instability")
    elif success_rate is None:
        decision = "hold"
        gating_result = "pending"
        reasons.append("eval_success_rate_missing")
    elif success_rate >= success_threshold:
        decision = "deploy_recommended"
        gating_result = "pass"
        reasons.append("eval_success_rate_meets_threshold")
    else:
        decision = "collect_more_data"
        gating_result = "fail"
        reasons.append("eval_success_rate_below_threshold")
    if warnings:
        reasons.append("training_monitor_warnings_present")

    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "decision": decision,
        "gating_result": gating_result,
        "success_rate": success_rate,
        "success_threshold": success_threshold,
        "warnings": warnings,
        "errors": errors,
        "reasons": reasons,
        "created_at": _now_iso(),
    }


def _annotation_phases(annotation: dict[str, Any]) -> tuple[str, ...] | None:
    l1 = annotation.get("l1_annotation")
    if isinstance(l1, dict) and isinstance(l1.get("phases"), list):
        return tuple(str(item) for item in l1["phases"])
    task = annotation.get("task")
    if isinstance(task, dict) and isinstance(task.get("phases"), list):
        return tuple(str(item) for item in task["phases"])
    return None


def _review_score_for_decision(decision: str) -> str:
    return "A" if decision == "accepted" else "B"


def _success_rate(eval_payload: dict[str, Any] | None) -> float | None:
    if not eval_payload:
        return None
    for key in ("success_rate", "eval_success", "eval_success_rate", "success"):
        if key in eval_payload:
            try:
                return float(eval_payload[key])
            except (TypeError, ValueError):
                return None
    return None


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _write_text_atomic(path, text)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_id(profile_name: str, policy_version: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{profile_name}_{policy_version}").strip("_")
    return f"{_utc_stamp()}_{slug}_{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
