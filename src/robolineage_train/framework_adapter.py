from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from glob import glob
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from robolineage_schemas.artifacts import write_validated_json_atomic

from .monitor import TrainingLogMonitor, TrainingMonitorAgent


PROFILE_SCHEMA_VERSION = "RoboLineage.framework_profile.v1"
SELECTED_SCHEMA_VERSION = "RoboLineage.selected_rollouts.v1"
RESULT_SCHEMA_VERSION = "RoboLineage.framework_run.v1"
_VALIDATED_ARTIFACT_SCHEMAS = {
    "training_status.json": "training_status",
    "training_result.json": "training_result",
}


@dataclass(frozen=True)
class FrameworkCommand:
    args: tuple[str, ...]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameworkOutputs:
    checkpoint_glob: str | None = None
    train_log: str | None = None
    eval_result: str | None = None


@dataclass(frozen=True)
class FrameworkStaging:
    selected_rollouts_file: str | None = None
    selected_rollouts_dir: str | None = None
    framework_input_dir: str | None = None


@dataclass(frozen=True)
class FrameworkExecution:
    train_launch_mode: str = "inline"
    terminal_command: tuple[str, ...] | None = None
    terminal_hold_open: bool = True
    tmux_session_name: str | None = None
    remote: "FrameworkRemoteExecution" = field(default_factory=lambda: FrameworkRemoteExecution())


@dataclass(frozen=True)
class FrameworkRemoteExecution:
    host: str | None = None
    repo_root: str | None = None
    dataset_dir: str | None = None
    checkpoint_dir: str | None = None
    work_dir: str | None = None
    train_log: str | None = None
    ssh_args: tuple[str, ...] = ()
    rsync_args: tuple[str, ...] = ("-az", "--delete")
    sync_checkpoints: bool = True


@dataclass(frozen=True)
class FrameworkProfile:
    name: str
    repo_root: Path
    dataset_command: FrameworkCommand | None
    train_command: FrameworkCommand | None
    eval_command: FrameworkCommand | None
    outputs: FrameworkOutputs
    staging: FrameworkStaging = field(default_factory=FrameworkStaging)
    execution: FrameworkExecution = field(default_factory=FrameworkExecution)
    dataset_adapter: dict[str, Any] = field(default_factory=dict)
    monitor: dict[str, Any] = field(default_factory=dict)
    log_patterns: dict[str, str] = field(default_factory=dict)
    adapter_version: str = "0.1"
    framework_type: str = "generic"


@dataclass(frozen=True)
class SelectedRollout:
    rollout_id: str
    rollout_dir: str
    decision: str
    data_use: tuple[str, ...]
    final_success: bool | None
    failure_candidate_count: int
    task_description: str


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: tuple[str, ...]
    cwd: str
    returncode: int
    log_path: Path
    parsed: dict[str, Any]
    tmux_session: str | None = None


@dataclass(frozen=True)
class FrameworkRunResult:
    framework: str
    framework_type: str
    adapter_version: str
    dataset_version: str
    policy_version: str
    staging_dir: Path
    selected_rollouts_file: Path
    selected_rollouts_dir: Path | None
    framework_input_dir: Path | None
    dataset_output_dir: Path
    dataset_adapter_plan_path: Path | None
    dataset_adapt_status_path: Path | None
    checkpoint_dir: Path
    checkpoint_path: Path | None
    eval_output_dir: Path | None
    training_status_path: Path | None
    training_result_path: Path
    eval_result_path: Path | None
    tmux_session: str | None = None


@dataclass(frozen=True)
class DatasetAdaptResult:
    framework: str
    framework_type: str
    adapter_version: str
    dataset_version: str
    policy_version: str
    staging_dir: Path
    selected_rollouts_file: Path
    selected_rollouts_dir: Path | None
    framework_input_dir: Path | None
    dataset_output_dir: Path
    dataset_adapter_plan_path: Path | None
    dataset_adapt_status_path: Path
    dataset_adapt_result_path: Path
    selected_rollout_count: int


@dataclass(frozen=True)
class _PreparedRun:
    workspace: Path
    staging_dir: Path
    dataset_output: Path
    checkpoint_dir: Path
    eval_output: Path
    context: dict[str, str]
    selected_file: Path
    selected_dir: Path | None
    framework_input_dir: Path | None
    dataset_adapter_plan_path: Path | None
    dataset_adapt_status_path: Path
    selected_rollout_count: int


class FrameworkAdapter:
    """Lightweight bridge to an existing training repository.

    RoboLineage owns selection, traceability and monitoring. The host repository owns
    dataset building, training and evaluation internals.
    """

    def __init__(
        self,
        profile: FrameworkProfile,
        *,
        training_monitor_llm_client: Any | None = None,
        enable_training_monitor_llm: bool | None = None,
    ) -> None:
        self.profile = profile
        self._training_monitor_llm_client = training_monitor_llm_client
        self._enable_training_monitor_llm = True if enable_training_monitor_llm is None else bool(enable_training_monitor_llm)
        self._enable_training_monitor_env_llm = (
            bool(os.environ.get("TRAINING_MONITOR_LLM_API_KEY"))
            if enable_training_monitor_llm is None
            else bool(enable_training_monitor_llm)
        )

    def run(
        self,
        *,
        rollouts_root: Path,
        workspace_dir: Path,
        dataset_version: str,
        policy_version: str,
        include_decisions: tuple[str, ...] = ("accepted",),
        include_rollout_ids: tuple[str, ...] | None = None,
        symlink_selected: bool | None = None,
    ) -> FrameworkRunResult:
        prepared = self._prepare_run(
            rollouts_root=rollouts_root,
            workspace_dir=workspace_dir,
            dataset_version=dataset_version,
            policy_version=policy_version,
            include_decisions=include_decisions,
            include_rollout_ids=include_rollout_ids,
            symlink_selected=symlink_selected,
        )
        command_results = self._run_dataset_adapt(prepared)
        return self._run_training_from_prepared(prepared, command_results)

    def adapt_dataset(
        self,
        *,
        rollouts_root: Path,
        workspace_dir: Path,
        dataset_version: str,
        policy_version: str,
        include_decisions: tuple[str, ...] = ("accepted",),
        include_rollout_ids: tuple[str, ...] | None = None,
        symlink_selected: bool | None = None,
    ) -> DatasetAdaptResult:
        prepared = self._prepare_run(
            rollouts_root=rollouts_root,
            workspace_dir=workspace_dir,
            dataset_version=dataset_version,
            policy_version=policy_version,
            include_decisions=include_decisions,
            include_rollout_ids=include_rollout_ids,
            symlink_selected=symlink_selected,
        )
        command_results = self._run_dataset_adapt(prepared)
        result_path = prepared.workspace / "dataset_adapt_result.json"
        result = DatasetAdaptResult(
            framework=self.profile.name,
            framework_type=self.profile.framework_type,
            adapter_version=self.profile.adapter_version,
            dataset_version=dataset_version,
            policy_version=policy_version,
            staging_dir=prepared.staging_dir,
            selected_rollouts_file=prepared.selected_file,
            selected_rollouts_dir=prepared.selected_dir,
            framework_input_dir=prepared.framework_input_dir,
            dataset_output_dir=prepared.dataset_output,
            dataset_adapter_plan_path=prepared.dataset_adapter_plan_path,
            dataset_adapt_status_path=prepared.dataset_adapt_status_path,
            dataset_adapt_result_path=result_path,
            selected_rollout_count=prepared.selected_rollout_count,
        )
        _write_json_atomic(
            result_path,
            {
                "schema_version": "RoboLineage.dataset_adapt_result.v1",
                **_jsonable_asdict(result),
                "commands": [_jsonable_asdict(item) for item in command_results],
                "created_at": _now_iso(),
            },
        )
        return result

    def run_training_only(
        self,
        *,
        workspace_dir: Path,
        dataset_version: str,
        policy_version: str,
    ) -> FrameworkRunResult:
        prepared = self._load_prepared_run(
            workspace_dir=workspace_dir,
            dataset_version=dataset_version,
            policy_version=policy_version,
        )
        adapt_status = _read_json(prepared.dataset_adapt_status_path) or {}
        if not _dataset_adapt_status_allows_training(adapt_status):
            raise RuntimeError(
                "dataset adaptation must complete before training; "
                f"current status={adapt_status.get('status') or 'missing'}"
            )
        return self._run_training_from_prepared(prepared, [])

    def _prepare_run(
        self,
        *,
        rollouts_root: Path,
        workspace_dir: Path,
        dataset_version: str,
        policy_version: str,
        include_decisions: tuple[str, ...],
        include_rollout_ids: tuple[str, ...] | None,
        symlink_selected: bool | None,
    ) -> _PreparedRun:
        workspace = Path(workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        staging_dir = workspace / "staging"
        dataset_output = workspace / "dataset"
        checkpoint_dir = workspace / "checkpoints" / policy_version
        eval_output = workspace / "eval"
        context = {
            "run_id": workspace.parent.name,
            "repo_root": str(self.profile.repo_root),
            "workspace_dir": str(workspace),
            "staging_dir": str(staging_dir),
            "rollouts_root": str(rollouts_root),
            "dataset_output": str(dataset_output),
            "checkpoint_dir": str(checkpoint_dir),
            "eval_output": str(eval_output),
            "dataset_version": dataset_version,
            "policy_version": policy_version,
        }
        selected_file = self._resolve_output_path(
            self.profile.staging.selected_rollouts_file or "{staging_dir}/selected_rollouts.json",
            context,
            base_dir=workspace,
        ) or (staging_dir / "selected_rollouts.json")
        if symlink_selected is None:
            symlink_selected = _needs_framework_input_staging(self.profile)
        selected_dir = None
        if symlink_selected:
            selected_dir = self._resolve_output_path(
                self.profile.staging.selected_rollouts_dir or "{staging_dir}/selected_rollouts",
                context,
                base_dir=workspace,
            )
        framework_input_template = self.profile.staging.framework_input_dir
        if framework_input_template is None:
            framework_input_template = "{selected_rollouts_dir}" if selected_dir is not None else "{dataset_output}"
        framework_input_dir = self._resolve_output_path(
            framework_input_template,
            {
                **context,
                "selected_rollouts_file": str(selected_file),
                "selected_rollouts_dir": str(selected_dir) if selected_dir is not None else "",
            },
            base_dir=workspace,
        )

        selected = write_selected_rollouts(
            rollouts_root=rollouts_root,
            output_path=selected_file,
            dataset_version=dataset_version,
            include_decisions=include_decisions,
            include_rollout_ids=include_rollout_ids,
            selected_dir=selected_dir,
        )
        if selected_dir is not None and framework_input_dir is not None and framework_input_dir != selected_dir:
            _write_selected_symlinks(selected, framework_input_dir)

        context.update({
            "selected_rollouts_file": str(selected_file),
            "selected_rollouts_dir": str(selected_dir) if selected_dir is not None else "",
            "framework_input_dir": str(framework_input_dir) if framework_input_dir is not None else "",
        })
        dataset_adapter_plan_path = self._write_dataset_adapter_plan(workspace, context)
        dataset_adapt_status_path = workspace / "dataset_adapt_status.json"
        _write_dataset_adapt_status(
            dataset_adapt_status_path,
            status="pending",
            profile=self.profile,
            context=context,
            selected_rollout_count=len(selected),
            adapter_plan_path=dataset_adapter_plan_path,
        )
        _write_json_atomic(
            workspace / "framework_context.json",
            {
                "schema_version": "RoboLineage.framework_context.v1",
                "context": context,
                "selected_rollout_count": len(selected),
                "selected_rollouts_file": str(selected_file),
                "selected_rollouts_dir": str(selected_dir) if selected_dir is not None else None,
                "framework_input_dir": str(framework_input_dir) if framework_input_dir is not None else None,
                "dataset_adapter_plan_path": str(dataset_adapter_plan_path) if dataset_adapter_plan_path is not None else None,
                "dataset_adapt_status_path": str(dataset_adapt_status_path),
                "created_at": _now_iso(),
            },
        )
        return _PreparedRun(
            workspace=workspace,
            staging_dir=staging_dir,
            dataset_output=dataset_output,
            checkpoint_dir=checkpoint_dir,
            eval_output=eval_output,
            context=context,
            selected_file=selected_file,
            selected_dir=selected_dir,
            framework_input_dir=framework_input_dir,
            dataset_adapter_plan_path=dataset_adapter_plan_path,
            dataset_adapt_status_path=dataset_adapt_status_path,
            selected_rollout_count=len(selected),
        )

    def _load_prepared_run(
        self,
        *,
        workspace_dir: Path,
        dataset_version: str,
        policy_version: str,
    ) -> _PreparedRun:
        workspace = Path(workspace_dir)
        payload = _read_json(workspace / "framework_context.json") or {}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        if not context:
            raise FileNotFoundError(f"framework_context.json not found in {workspace}")
        context = {str(key): str(value) for key, value in context.items()}
        context.setdefault("run_id", workspace.parent.name)
        context["dataset_version"] = dataset_version
        context["policy_version"] = policy_version
        selected_file = Path(str(payload.get("selected_rollouts_file") or context["selected_rollouts_file"]))
        selected_dir = Path(str(payload["selected_rollouts_dir"])) if payload.get("selected_rollouts_dir") else None
        framework_input_dir = Path(str(payload["framework_input_dir"])) if payload.get("framework_input_dir") else None
        plan_path = Path(str(payload["dataset_adapter_plan_path"])) if payload.get("dataset_adapter_plan_path") else None
        status_path = Path(str(payload.get("dataset_adapt_status_path") or workspace / "dataset_adapt_status.json"))
        return _PreparedRun(
            workspace=workspace,
            staging_dir=Path(context["staging_dir"]),
            dataset_output=Path(context["dataset_output"]),
            checkpoint_dir=Path(context["checkpoint_dir"]),
            eval_output=Path(context["eval_output"]),
            context=context,
            selected_file=selected_file,
            selected_dir=selected_dir,
            framework_input_dir=framework_input_dir,
            dataset_adapter_plan_path=plan_path,
            dataset_adapt_status_path=status_path,
            selected_rollout_count=int(payload.get("selected_rollout_count") or 0),
        )

    def _run_dataset_adapt(self, prepared: _PreparedRun) -> list[CommandResult]:
        command_results: list[CommandResult] = []
        try:
            if self.profile.dataset_command is not None:
                prepared.dataset_output.mkdir(parents=True, exist_ok=True)
                _write_dataset_adapt_status(
                    prepared.dataset_adapt_status_path,
                    status="running",
                    profile=self.profile,
                    context=prepared.context,
                    selected_rollout_count=prepared.selected_rollout_count,
                    adapter_plan_path=prepared.dataset_adapter_plan_path,
                )
                command_results.append(
                    self._run_command("dataset", self.profile.dataset_command, prepared.context, prepared.workspace)
                )
                _write_dataset_adapt_status(
                    prepared.dataset_adapt_status_path,
                    status="completed",
                    profile=self.profile,
                    context=prepared.context,
                    selected_rollout_count=prepared.selected_rollout_count,
                    adapter_plan_path=prepared.dataset_adapter_plan_path,
                    adapter_report=_find_dataset_adapter_report(prepared.dataset_output),
                )
            else:
                unresolved_reason = _dataset_adapter_unresolved_reason(self.profile)
                if unresolved_reason is not None:
                    raise RuntimeError(
                        "dataset adapter is unresolved; discovery could not infer a concrete "
                        "training dataset contract or conversion command"
                    )
                _write_dataset_adapt_status(
                    prepared.dataset_adapt_status_path,
                    status="skipped",
                    profile=self.profile,
                    context=prepared.context,
                    selected_rollout_count=prepared.selected_rollout_count,
                    adapter_plan_path=prepared.dataset_adapter_plan_path,
                    reason="profile_has_no_dataset_command",
                )
            return command_results
        except subprocess.CalledProcessError as exc:
            _write_dataset_adapt_status(
                prepared.dataset_adapt_status_path,
                status="failed",
                profile=self.profile,
                context=prepared.context,
                selected_rollout_count=prepared.selected_rollout_count,
                adapter_plan_path=prepared.dataset_adapter_plan_path,
                error=str(exc),
            )
            raise
        except Exception as exc:
            _write_dataset_adapt_status(
                prepared.dataset_adapt_status_path,
                status="failed",
                profile=self.profile,
                context=prepared.context,
                selected_rollout_count=prepared.selected_rollout_count,
                adapter_plan_path=prepared.dataset_adapter_plan_path,
                reason=_dataset_adapter_unresolved_reason(self.profile),
                error=str(exc),
            )
            raise

    def _run_training_from_prepared(
        self,
        prepared: _PreparedRun,
        command_results: list[CommandResult],
    ) -> FrameworkRunResult:
        training_status_path = prepared.workspace / "training_status.json"
        context = dict(prepared.context)
        dataset_version = context["dataset_version"]
        policy_version = context["policy_version"]
        try:
            prepared.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            train_result: CommandResult | None = None
            if self.profile.train_command is not None:
                train_result = self._run_command("train", self.profile.train_command, context, prepared.workspace)
                command_results.append(train_result)

            checkpoint_path = self._find_checkpoint(prepared.checkpoint_dir, context)
            if checkpoint_path is not None:
                context["checkpoint_path"] = str(checkpoint_path)

            eval_result_path: Path | None = None
            if self.profile.eval_command is not None:
                prepared.eval_output.mkdir(parents=True, exist_ok=True)
                eval_command_result = self._run_command(
                    "eval",
                    self.profile.eval_command,
                    context,
                    prepared.workspace,
                )
                command_results.append(eval_command_result)
                eval_result_path = self._resolve_output_path(
                    self.profile.outputs.eval_result,
                    context,
                    base_dir=self.profile.repo_root,
                )
        except subprocess.CalledProcessError as exc:
            _write_json_atomic(
                training_status_path,
                {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "status": "failed",
                    "framework": self.profile.name,
                    "dataset_version": dataset_version,
                    "policy_version": policy_version,
                    "selected_rollout_count": prepared.selected_rollout_count,
                    "failed_command": list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd),
                    "returncode": exc.returncode,
                    "updated_at": _now_iso(),
                },
            )
            raise
        except Exception as exc:
            _write_json_atomic(
                training_status_path,
                {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "status": "failed",
                    "framework": self.profile.name,
                    "dataset_version": dataset_version,
                    "policy_version": policy_version,
                    "selected_rollout_count": prepared.selected_rollout_count,
                    "error": str(exc),
                    "updated_at": _now_iso(),
                },
            )
            raise

        train_parsed = train_result.parsed if train_result is not None else {}
        monitor_understanding = self._training_monitor_understanding(
            train_result=train_result,
            train_parsed=train_parsed,
            prepared=prepared,
        )
        tmux_session = train_result.tmux_session if train_result is not None else None
        monitor_status = str(train_parsed.get("status") or "completed")
        status = monitor_status if monitor_status in {"failed", "unstable"} else "completed"
        _write_json_atomic(
            training_status_path,
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": status,
                "framework": self.profile.name,
                "dataset_version": dataset_version,
                "policy_version": policy_version,
                "selected_rollout_count": prepared.selected_rollout_count,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "tmux_session": tmux_session,
                "metrics": train_parsed,
                "llm_understanding": monitor_understanding,
                "updated_at": _now_iso(),
            },
        )

        result_path = prepared.workspace / "training_result.json"
        result = FrameworkRunResult(
            framework=self.profile.name,
            framework_type=self.profile.framework_type,
            adapter_version=self.profile.adapter_version,
            dataset_version=dataset_version,
            policy_version=policy_version,
            staging_dir=prepared.staging_dir,
            selected_rollouts_file=prepared.selected_file,
            selected_rollouts_dir=prepared.selected_dir,
            framework_input_dir=prepared.framework_input_dir,
            dataset_output_dir=prepared.dataset_output,
            dataset_adapter_plan_path=prepared.dataset_adapter_plan_path,
            dataset_adapt_status_path=prepared.dataset_adapt_status_path,
            checkpoint_dir=prepared.checkpoint_dir,
            checkpoint_path=checkpoint_path,
            eval_output_dir=prepared.eval_output if self.profile.eval_command is not None else None,
            training_status_path=training_status_path,
            training_result_path=result_path,
            eval_result_path=eval_result_path if eval_result_path and eval_result_path.exists() else None,
            tmux_session=tmux_session,
        )
        _write_json_atomic(
            result_path,
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                **_jsonable_asdict(result),
                "commands": [_jsonable_asdict(item) for item in command_results],
                "selected_rollout_count": prepared.selected_rollout_count,
                "created_at": _now_iso(),
            },
        )
        return result

    def _training_monitor_understanding(
        self,
        *,
        train_result: CommandResult | None,
        train_parsed: dict[str, Any],
        prepared: _PreparedRun,
    ) -> dict[str, Any]:
        log_text = ""
        log_path = None
        if train_result is not None:
            log_path = train_result.log_path
            if train_result.log_path.exists():
                log_text = train_result.log_path.read_text(encoding="utf-8")
        try:
            result = TrainingMonitorAgent(
                llm_client=self._training_monitor_llm_client,
                enable_llm_understanding=self._enable_training_monitor_llm,
                enable_env_llm=self._enable_training_monitor_env_llm,
            ).analyze(
                log_text,
                patterns=self.profile.log_patterns,
                monitor_spec=self.profile.monitor,
                context={
                    "framework": self.profile.name,
                    "framework_type": self.profile.framework_type,
                    "policy_version": prepared.context.get("policy_version"),
                    "dataset_version": prepared.context.get("dataset_version"),
                    "log_path": str(log_path) if log_path is not None else None,
                },
                output_dir=prepared.workspace,
                deterministic_report=train_parsed,
            )
            return dict(result.report.get("llm_understanding") or {})
        except Exception as exc:
            return {
                "status": "failed",
                "model": None,
                "summary": "Training monitor LLM understanding failed before writing artifacts.",
                "error": str(exc),
            }

    def _write_dataset_adapter_plan(self, workspace: Path, context: dict[str, str]) -> Path | None:
        if not self.profile.dataset_adapter:
            return None
        path = workspace / "dataset_adapter_plan.json"
        payload = {
            **self.profile.dataset_adapter,
            "rendered": {
                "selected_rollouts_file": context["selected_rollouts_file"],
                "selected_rollouts_dir": context["selected_rollouts_dir"] or None,
                "framework_input_dir": context["framework_input_dir"] or None,
                "dataset_output": context["dataset_output"],
            },
        }
        _write_json_atomic(path, payload)
        return path

    def _run_command(
        self,
        name: str,
        command: FrameworkCommand,
        context: dict[str, str],
        workspace: Path,
    ) -> CommandResult:
        if name == "train" and self.profile.execution.train_launch_mode == "remote_tmux":
            return self._run_command_remote_tmux(name, command, context, workspace)
        if name == "train" and self.profile.execution.train_launch_mode == "tmux":
            return self._run_command_tmux(name, command, context, workspace)
        if name == "train" and self.profile.execution.train_launch_mode == "external_terminal":
            return self._run_command_external_terminal(name, command, context, workspace)
        return self._run_command_inline(name, command, context, workspace)

    def _run_command_inline(
        self,
        name: str,
        command: FrameworkCommand,
        context: dict[str, str],
        workspace: Path,
    ) -> CommandResult:
        args = tuple(_render_template(part, context) for part in command.args)
        cwd = Path(_render_template(command.cwd, context)) if command.cwd else self.profile.repo_root
        log_path = workspace / f"{name}_command.log"
        env = _command_env(command, context)
        status_path = workspace / "training_status.json"
        _write_running_status(status_path, self.profile.name, context, name, log_path, {})
        proc = subprocess.Popen(
            list(args),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        lines: list[str] = []
        last_update = 0.0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            if proc.stdout is not None:
                for line in proc.stdout:
                    lines.append(line)
                    log_file.write(line)
                    log_file.flush()
                    now = time.monotonic()
                    if now - last_update >= 1.0:
                        parsed_partial = parse_training_log(
                            "".join(lines),
                            self.profile.log_patterns,
                            self.profile.monitor,
                        )
                        _write_running_status(status_path, self.profile.name, context, name, log_path, parsed_partial)
                        last_update = now
        returncode = proc.wait()
        log_text = "".join(lines)
        if returncode != 0:
            raise subprocess.CalledProcessError(
                returncode,
                list(args),
                output=log_text,
                stderr="",
            )
        parsed = parse_training_log(log_text, self.profile.log_patterns, self.profile.monitor)
        train_log = self._resolve_output_path(self.profile.outputs.train_log, context, base_dir=cwd)
        if name == "train" and train_log is not None and train_log.exists():
            parsed.update(
                parse_training_log(
                    train_log.read_text(encoding="utf-8"),
                    self.profile.log_patterns,
                    self.profile.monitor,
                )
            )
        return CommandResult(
            name=name,
            command=args,
            cwd=str(cwd),
            returncode=returncode,
            log_path=log_path,
            parsed=parsed,
        )

    def _run_command_tmux(
        self,
        name: str,
        command: FrameworkCommand,
        context: dict[str, str],
        workspace: Path,
    ) -> CommandResult:
        tmux = shutil.which("tmux")
        if not tmux:
            raise FileNotFoundError("tmux command not found; install tmux or use inline training launch mode")

        args = tuple(_render_template(part, context) for part in command.args)
        cwd = Path(_render_template(command.cwd, context)) if command.cwd else self.profile.repo_root
        log_path = workspace / f"{name}_command.log"
        exit_code_path = workspace / f"{name}_exit_code.txt"
        script_path = workspace / f"{name}_tmux.sh"
        env = _command_env(command, context)
        status_path = workspace / "training_status.json"
        session_name = _tmux_session_name(self.profile.execution, context)

        workspace.mkdir(parents=True, exist_ok=True)
        if exit_code_path.exists():
            exit_code_path.unlink()
        _write_external_terminal_script(
            script_path=script_path,
            cwd=cwd,
            args=args,
            env=env,
            log_path=log_path,
            exit_code_path=exit_code_path,
            hold_open=False,
        )
        _write_running_status(
            status_path,
            self.profile.name,
            context,
            name,
            log_path,
            {"launch_mode": "tmux", "tmux_session": session_name},
        )

        launcher = [tmux, "new-session", "-d", "-s", session_name, "bash", str(script_path)]
        launch_proc = subprocess.run(
            launcher,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
        if launch_proc.returncode != 0:
            raise subprocess.CalledProcessError(
                launch_proc.returncode,
                launcher,
                output=launch_proc.stdout,
                stderr="tmux failed to launch training session",
            )

        last_update = 0.0
        while not exit_code_path.exists():
            now = time.monotonic()
            if now - last_update >= 1.0:
                log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
                parsed_partial = parse_training_log(log_text, self.profile.log_patterns, self.profile.monitor)
                parsed_partial["launch_mode"] = "tmux"
                parsed_partial["tmux_session"] = session_name
                _write_running_status(status_path, self.profile.name, context, name, log_path, parsed_partial)
                last_update = now
            time.sleep(0.5)

        try:
            returncode = int(exit_code_path.read_text(encoding="utf-8").strip())
        except ValueError:
            returncode = 1
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        if returncode != 0:
            raise subprocess.CalledProcessError(
                returncode,
                list(args),
                output=log_text,
                stderr="",
            )
        parsed = parse_training_log(log_text, self.profile.log_patterns, self.profile.monitor)
        parsed["launch_mode"] = "tmux"
        parsed["tmux_session"] = session_name
        train_log = self._resolve_output_path(self.profile.outputs.train_log, context, base_dir=cwd)
        if name == "train" and train_log is not None and train_log.exists():
            parsed.update(
                parse_training_log(
                    train_log.read_text(encoding="utf-8"),
                    self.profile.log_patterns,
                    self.profile.monitor,
                )
            )
            parsed["launch_mode"] = "tmux"
            parsed["tmux_session"] = session_name
        return CommandResult(
            name=name,
            command=args,
            cwd=str(cwd),
            returncode=returncode,
            log_path=log_path,
            parsed=parsed,
            tmux_session=session_name,
        )

    def _run_command_remote_tmux(
        self,
        name: str,
        command: FrameworkCommand,
        context: dict[str, str],
        workspace: Path,
    ) -> CommandResult:
        ssh = shutil.which("ssh")
        rsync = shutil.which("rsync")
        if not ssh:
            raise FileNotFoundError("ssh command not found; install OpenSSH client for remote_tmux")
        if not rsync:
            raise FileNotFoundError("rsync command not found; install rsync for remote_tmux dataset sync")

        remote = self.profile.execution.remote
        if not remote.host:
            raise ValueError("execution.remote.host is required for remote_tmux")

        remote_context = _remote_context(self.profile.execution, context, self.profile.repo_root)
        args = tuple(_render_template(part, remote_context) for part in command.args)
        cwd = _render_template(command.cwd, remote_context) if command.cwd else remote_context["repo_root"]
        env = _command_env(command, remote_context)
        log_path = workspace / f"{name}_command.log"
        exit_code_path = workspace / f"{name}_exit_code.txt"
        status_path = workspace / "training_status.json"
        session_name = _tmux_session_name(self.profile.execution, remote_context)
        remote_log_path = remote_context["remote_train_log"]
        remote_exit_code_path = remote_context["remote_exit_code_path"]
        remote_script_path = remote_context["remote_train_script"]

        workspace.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if exit_code_path.exists():
            exit_code_path.unlink()
        log_path.write_text("", encoding="utf-8")

        parsed_base = {
            "launch_mode": "remote_tmux",
            "tmux_session": session_name,
            "remote_host": remote.host,
            "remote_dataset_dir": remote_context["dataset_output"],
            "remote_checkpoint_dir": remote_context["checkpoint_dir"],
            "remote_train_log": remote_log_path,
        }
        _write_running_status(status_path, self.profile.name, context, name, log_path, parsed_base)

        ssh_args = list(remote.ssh_args)
        _run_remote_shell(
            ssh,
            remote.host,
            ssh_args,
            "mkdir -p "
            + " ".join(
                _quote_remote_path(path)
                for path in (
                    remote_context["dataset_output"],
                    remote_context["checkpoint_dir"],
                    remote_context["remote_work_dir"],
                    str(Path(remote_log_path).parent),
                )
            ),
        )
        _sync_to_remote(
            rsync,
            source=Path(context["dataset_output"]),
            host=remote.host,
            destination=remote_context["dataset_output"],
            ssh_args=ssh_args,
            rsync_args=list(remote.rsync_args),
        )
        _write_remote_file(
            ssh,
            remote.host,
            ssh_args,
            remote_script_path,
            _remote_training_script(
                cwd=cwd,
                args=args,
                env=env,
                log_path=remote_log_path,
                exit_code_path=remote_exit_code_path,
            ),
        )
        _run_remote_shell(
            ssh,
            remote.host,
            ssh_args,
            "rm -f "
            + _quote_remote_path(remote_exit_code_path)
            + " && tmux new-session -d -s "
            + shlex.quote(session_name)
            + " bash "
            + _quote_remote_path(remote_script_path),
        )

        last_update = 0.0
        returncode: int | None = None
        while returncode is None:
            now = time.monotonic()
            if now - last_update >= 1.0:
                remote_log_text = _read_remote_text(
                    ssh,
                    remote.host,
                    ssh_args,
                    remote_log_path,
                    missing_ok=True,
                )
                log_path.write_text(remote_log_text, encoding="utf-8")
                parsed_partial = parse_training_log(remote_log_text, self.profile.log_patterns, self.profile.monitor)
                parsed_partial.update(parsed_base)
                _write_running_status(status_path, self.profile.name, context, name, log_path, parsed_partial)
                exit_text = _read_remote_text(
                    ssh,
                    remote.host,
                    ssh_args,
                    remote_exit_code_path,
                    missing_ok=True,
                ).strip()
                if exit_text:
                    try:
                        returncode = int(exit_text.splitlines()[-1])
                    except ValueError:
                        returncode = 1
                last_update = now
            time.sleep(0.5)

        log_text = _read_remote_text(ssh, remote.host, ssh_args, remote_log_path, missing_ok=True)
        log_path.write_text(log_text, encoding="utf-8")
        exit_code_path.write_text(f"{returncode}\n", encoding="utf-8")
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, list(args), output=log_text, stderr="")

        if remote.sync_checkpoints:
            _sync_from_remote(
                rsync,
                host=remote.host,
                source=remote_context["checkpoint_dir"],
                destination=Path(context["checkpoint_dir"]),
                ssh_args=ssh_args,
                rsync_args=list(remote.rsync_args),
            )
        parsed = parse_training_log(log_text, self.profile.log_patterns, self.profile.monitor)
        parsed.update(parsed_base)
        return CommandResult(
            name=name,
            command=args,
            cwd=cwd,
            returncode=returncode,
            log_path=log_path,
            parsed=parsed,
            tmux_session=session_name,
        )

    def _run_command_external_terminal(
        self,
        name: str,
        command: FrameworkCommand,
        context: dict[str, str],
        workspace: Path,
    ) -> CommandResult:
        args = tuple(_render_template(part, context) for part in command.args)
        cwd = Path(_render_template(command.cwd, context)) if command.cwd else self.profile.repo_root
        log_path = workspace / f"{name}_command.log"
        exit_code_path = workspace / f"{name}_exit_code.txt"
        script_path = workspace / f"{name}_external_terminal.sh"
        env = _command_env(command, context)
        status_path = workspace / "training_status.json"

        workspace.mkdir(parents=True, exist_ok=True)
        if exit_code_path.exists():
            exit_code_path.unlink()
        _write_external_terminal_script(
            script_path=script_path,
            cwd=cwd,
            args=args,
            env=env,
            log_path=log_path,
            exit_code_path=exit_code_path,
            hold_open=self.profile.execution.terminal_hold_open,
        )
        _write_running_status(status_path, self.profile.name, context, name, log_path, {})

        launcher = _terminal_launcher_args(script_path, self.profile.execution)
        launcher_proc = subprocess.Popen(
            launcher,
            cwd=str(cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )

        last_update = 0.0
        launched_at = time.monotonic()
        while not exit_code_path.exists():
            launcher_returncode = launcher_proc.poll()
            if launcher_returncode not in (None, 0):
                raise subprocess.CalledProcessError(
                    launcher_returncode,
                    launcher,
                    output=log_path.read_text(encoding="utf-8") if log_path.exists() else "",
                    stderr="external terminal launcher exited before training finished",
                )
            if (
                launcher_returncode == 0
                and time.monotonic() - launched_at > 15.0
                and (not log_path.exists() or log_path.stat().st_size == 0)
            ):
                raise RuntimeError(
                    "external terminal launcher exited but the training script did not write a log; "
                    "check execution.terminal_command and desktop DISPLAY permissions"
                )
            now = time.monotonic()
            if now - last_update >= 1.0:
                log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
                parsed_partial = parse_training_log(log_text, self.profile.log_patterns, self.profile.monitor)
                parsed_partial["launch_mode"] = "external_terminal"
                _write_running_status(status_path, self.profile.name, context, name, log_path, parsed_partial)
                last_update = now
            time.sleep(0.5)

        try:
            returncode = int(exit_code_path.read_text(encoding="utf-8").strip())
        except ValueError:
            returncode = 1
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        if returncode != 0:
            raise subprocess.CalledProcessError(
                returncode,
                list(args),
                output=log_text,
                stderr="",
            )
        parsed = parse_training_log(log_text, self.profile.log_patterns, self.profile.monitor)
        parsed["launch_mode"] = "external_terminal"
        train_log = self._resolve_output_path(self.profile.outputs.train_log, context, base_dir=cwd)
        if name == "train" and train_log is not None and train_log.exists():
            parsed.update(
                parse_training_log(
                    train_log.read_text(encoding="utf-8"),
                    self.profile.log_patterns,
                    self.profile.monitor,
                )
            )
            parsed["launch_mode"] = "external_terminal"
        return CommandResult(
            name=name,
            command=args,
            cwd=str(cwd),
            returncode=returncode,
            log_path=log_path,
            parsed=parsed,
        )

    def _find_checkpoint(self, checkpoint_dir: Path, context: dict[str, str]) -> Path | None:
        checkpoint_glob = self.profile.outputs.checkpoint_glob
        if checkpoint_glob:
            pattern = _render_template(checkpoint_glob, context)
            pattern_path = Path(pattern)
            if pattern_path.is_absolute():
                matches = sorted((Path(item) for item in glob(pattern)), key=lambda path: path.stat().st_mtime)
            else:
                matches = sorted(self.profile.repo_root.glob(pattern), key=lambda path: path.stat().st_mtime)
            if matches:
                return matches[-1]
        matches = sorted(checkpoint_dir.glob("*"), key=lambda path: path.stat().st_mtime)
        return matches[-1] if matches else None

    @staticmethod
    def _resolve_output_path(
        value: str | None,
        context: dict[str, str],
        *,
        base_dir: Path,
    ) -> Path | None:
        if not value:
            return None
        path = Path(_render_template(value, context))
        return path if path.is_absolute() else base_dir / path


def load_framework_profile(path: Path) -> FrameworkProfile:
    profile_path = Path(path)
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("framework profile must be a mapping")
    repo_root = Path(str(raw.get("repo_root") or ".")).expanduser()
    if not repo_root.is_absolute():
        repo_root = profile_path.parent / repo_root
    repo_root = repo_root.resolve()
    return FrameworkProfile(
        name=str(raw["name"]),
        repo_root=repo_root,
        dataset_command=_command_or_none(raw.get("dataset_command")),
        train_command=_command_or_none(raw.get("train_command")),
        eval_command=_command_or_none(raw.get("eval_command")),
        outputs=_outputs(raw.get("outputs") or {}),
        staging=_staging(raw.get("staging") or {}),
        execution=_execution(raw.get("execution") or {}),
        dataset_adapter=dict(raw.get("dataset_adapter") or {}),
        monitor=dict(raw.get("monitor") or {}),
        log_patterns={str(k): str(v) for k, v in (raw.get("log_patterns") or {}).items()},
        adapter_version=str(raw.get("adapter_version") or "0.1"),
        framework_type=str(raw.get("framework_type") or "generic"),
    )


def write_selected_rollouts(
    *,
    rollouts_root: Path,
    output_path: Path,
    dataset_version: str,
    include_decisions: tuple[str, ...] = ("accepted",),
    include_rollout_ids: tuple[str, ...] | None = None,
    selected_dir: Path | None = None,
) -> list[SelectedRollout]:
    rollouts = select_rollouts(
        rollouts_root=rollouts_root,
        include_decisions=include_decisions,
        include_rollout_ids=include_rollout_ids,
    )
    payload = {
        "schema_version": SELECTED_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "created_at": _now_iso(),
        "selected_rollout_count": len(rollouts),
        "selected_rollouts": [_jsonable_asdict(item) for item in rollouts],
    }
    _write_json_atomic(output_path, payload)
    if selected_dir is not None:
        _write_selected_symlinks(rollouts, selected_dir)
    return rollouts


def select_rollouts(
    *,
    rollouts_root: Path,
    include_decisions: tuple[str, ...] = ("accepted",),
    include_rollout_ids: tuple[str, ...] | None = None,
) -> list[SelectedRollout]:
    allowed = set(include_decisions)
    allowed_rollouts = set(include_rollout_ids) if include_rollout_ids is not None else None
    rows: list[SelectedRollout] = []
    for rollout_dir in sorted(Path(rollouts_root).iterdir()):
        if not rollout_dir.is_dir():
            continue
        if allowed_rollouts is not None and rollout_dir.name not in allowed_rollouts:
            continue
        admission = _read_json(rollout_dir / "dataset_admission.json")
        if not admission:
            continue
        decision = str(admission.get("decision") or "")
        accepted_for_training = admission.get("accepted_for_training")
        if isinstance(accepted_for_training, bool):
            if not accepted_for_training:
                continue
        elif decision not in allowed:
            continue
        summary = _read_json(rollout_dir / "rollout_summary.json") or {}
        failures = _read_json(rollout_dir / "failure_analysis.json") or {}
        rows.append(
            SelectedRollout(
                rollout_id=rollout_dir.name,
                rollout_dir=str(rollout_dir.resolve()),
                decision=decision,
                data_use=tuple(str(item) for item in admission.get("data_use") or ()),
                final_success=_bool_or_none(summary.get("final_success", summary.get("success_likely"))),
                failure_candidate_count=int(failures.get("candidate_count") or 0),
                task_description=str(summary.get("task_description") or admission.get("task_description") or ""),
            )
        )
    return rows


def parse_training_log(
    text: str,
    patterns: dict[str, str],
    monitor_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = TrainingLogMonitor(patterns, monitor_spec).parse(text)
    out = report.to_dict()
    out.update(report.metrics)
    return out


def _command_or_none(raw: Any) -> FrameworkCommand | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("framework command must be a mapping")
    args = raw.get("args")
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("framework command args must be a list of strings")
    env = raw.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("framework command env must be a mapping")
    cwd = raw.get("cwd")
    return FrameworkCommand(
        args=tuple(args),
        cwd=str(cwd) if cwd is not None else None,
        env={str(k): str(v) for k, v in env.items()},
    )


def _outputs(raw: dict[str, Any]) -> FrameworkOutputs:
    return FrameworkOutputs(
        checkpoint_glob=_optional_str(raw.get("checkpoint_glob")),
        train_log=_optional_str(raw.get("train_log")),
        eval_result=_optional_str(raw.get("eval_result")),
    )


def _staging(raw: dict[str, Any]) -> FrameworkStaging:
    return FrameworkStaging(
        selected_rollouts_file=_optional_str(raw.get("selected_rollouts_file")),
        selected_rollouts_dir=_optional_str(raw.get("selected_rollouts_dir")),
        framework_input_dir=_optional_str(raw.get("framework_input_dir")),
    )


def _execution(raw: dict[str, Any]) -> FrameworkExecution:
    mode = str(raw.get("train_launch_mode") or "inline")
    if mode not in {"inline", "external_terminal", "tmux", "remote_tmux"}:
        raise ValueError(f"unsupported train_launch_mode: {mode}")
    terminal_command = raw.get("terminal_command")
    terminal_parts: tuple[str, ...] | None = None
    if terminal_command is not None:
        if isinstance(terminal_command, str):
            terminal_parts = tuple(shlex.split(terminal_command))
        elif isinstance(terminal_command, list) and all(isinstance(item, str) for item in terminal_command):
            terminal_parts = tuple(terminal_command)
        else:
            raise ValueError("execution.terminal_command must be a string or list of strings")
    return FrameworkExecution(
        train_launch_mode=mode,
        terminal_command=terminal_parts,
        terminal_hold_open=bool(raw.get("terminal_hold_open", True)),
        tmux_session_name=_optional_str(raw.get("tmux_session_name")),
        remote=_remote_execution(raw.get("remote") or {}),
    )


def _remote_execution(raw: Any) -> FrameworkRemoteExecution:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("execution.remote must be a mapping")
    return FrameworkRemoteExecution(
        host=_optional_str(raw.get("host")),
        repo_root=_optional_str(raw.get("repo_root")),
        dataset_dir=_optional_str(raw.get("dataset_dir")),
        checkpoint_dir=_optional_str(raw.get("checkpoint_dir")),
        work_dir=_optional_str(raw.get("work_dir")),
        train_log=_optional_str(raw.get("train_log")),
        ssh_args=_string_tuple(raw.get("ssh_args")),
        rsync_args=_string_tuple(raw.get("rsync_args")) or ("-az", "--delete"),
        sync_checkpoints=bool(raw.get("sync_checkpoints", True)),
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("expected a string or list of strings")


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _render_template(value: str | None, context: dict[str, str]) -> str:
    return "" if value is None else value.format(**context)


def _needs_framework_input_staging(profile: FrameworkProfile) -> bool:
    if profile.staging.framework_input_dir is not None or profile.staging.selected_rollouts_dir is not None:
        return True
    commands = (profile.dataset_command, profile.train_command, profile.eval_command)
    return any(_command_uses_placeholder(command, "framework_input_dir") for command in commands) or any(
        _command_uses_placeholder(command, "selected_rollouts_dir") for command in commands
    )


def _command_uses_placeholder(command: FrameworkCommand | None, placeholder: str) -> bool:
    if command is None:
        return False
    token = "{" + placeholder + "}"
    if any(token in part for part in command.args):
        return True
    if command.cwd and token in command.cwd:
        return True
    return any(token in value for value in command.env.values())


def _command_env(command: FrameworkCommand, context: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ROBOLINEAGE_SELECTED_ROLLOUTS": context["selected_rollouts_file"],
            "ROBOLINEAGE_SELECTED_ROLLOUTS_DIR": context["selected_rollouts_dir"],
            "ROBOLINEAGE_FRAMEWORK_INPUT_DIR": context["framework_input_dir"],
            "ROBOLINEAGE_STAGING_DIR": context["staging_dir"],
            "ROBOLINEAGE_DATASET_OUTPUT": context["dataset_output"],
            "ROBOLINEAGE_CHECKPOINT_DIR": context["checkpoint_dir"],
            "ROBOLINEAGE_EVAL_OUTPUT": context["eval_output"],
            "ROBOLINEAGE_DATASET_VERSION": context["dataset_version"],
            "ROBOLINEAGE_POLICY_VERSION": context["policy_version"],
        }
    )
    env.update({key: _render_template(value, context) for key, value in command.env.items()})
    return env


def _tmux_session_name(execution: FrameworkExecution, context: dict[str, str]) -> str:
    template = execution.tmux_session_name or "robolineage_train_{policy_version}_{dataset_version}"
    rendered = _render_template(template, context)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in rendered)
    return safe[:80] or "robolineage_train"


def _remote_context(
    execution: FrameworkExecution,
    context: dict[str, str],
    default_repo_root: Path,
) -> dict[str, str]:
    remote = execution.remote
    rendered: dict[str, str] = dict(context)
    rendered.setdefault("run_id", Path(context.get("workspace_dir", "")).parent.name or "run")
    run_id = rendered["run_id"]
    base = f"~/robolineage_training/runs/{run_id}"
    remote_work_dir = _render_template(remote.work_dir or base, rendered)
    rendered["repo_root"] = _render_template(remote.repo_root or str(default_repo_root), rendered)
    rendered["remote_work_dir"] = remote_work_dir
    rendered["dataset_output"] = _render_template(remote.dataset_dir or f"{remote_work_dir}/dataset", rendered)
    rendered["framework_input_dir"] = rendered["dataset_output"]
    rendered["checkpoint_dir"] = _render_template(remote.checkpoint_dir or f"{remote_work_dir}/checkpoints", rendered)
    rendered["remote_train_log"] = _render_template(remote.train_log or f"{rendered['checkpoint_dir']}/training.log", rendered)
    rendered["remote_exit_code_path"] = f"{remote_work_dir}/train_exit_code.txt"
    rendered["remote_train_script"] = f"{remote_work_dir}/train_remote_tmux.sh"
    return rendered


def _remote_training_script(
    *,
    cwd: str,
    args: tuple[str, ...],
    env: dict[str, str],
    log_path: str,
    exit_code_path: str,
) -> str:
    env_lines = [
        f"export {key}={shlex.quote(value)}"
        for key, value in sorted(env.items())
        if _should_export_to_terminal(key, value)
    ]
    command_array = " ".join(shlex.quote(item) for item in args)
    quoted_cwd = _quote_remote_path(cwd)
    quoted_log_path = _quote_remote_path(log_path)
    quoted_log_parent = _quote_remote_path(str(Path(log_path).parent))
    quoted_exit_code_path = _quote_remote_path(exit_code_path)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set +e",
            f"cd {quoted_cwd} || exit 127",
            *env_lines,
            f"mkdir -p {quoted_log_parent}",
            f": > {quoted_log_path}",
            f"cmd=({command_array})",
            "{",
            "  echo \"[RoboLineage] remote tmux training started\"",
            f"  echo \"[RoboLineage] cwd={shlex.quote(cwd)}\"",
            "  printf '[RoboLineage] command:'",
            "  printf ' %q' \"${cmd[@]}\"",
            "  echo",
            "  \"${cmd[@]}\"",
            "  cmd_status=$?",
            "  echo \"[RoboLineage] command_exit_code=${cmd_status}\"",
            "  exit \"${cmd_status}\"",
            f"}} 2>&1 | tee -a {quoted_log_path}",
            "cmd_status=${PIPESTATUS[0]}",
            f"tmp={quoted_exit_code_path}.tmp",
            "printf '%s\\n' \"${cmd_status}\" > \"${tmp}\"",
            f"mv \"${{tmp}}\" {quoted_exit_code_path}",
            "exit \"${cmd_status}\"",
            "",
        ]
    )


def _run_remote_shell(
    ssh: str,
    host: str,
    ssh_args: list[str],
    command: str,
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [ssh, *ssh_args, host, "bash", "-lc", command],
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, [ssh, *ssh_args, host, command], output=proc.stdout)
    return proc


def _write_remote_file(ssh: str, host: str, ssh_args: list[str], path: str, text: str) -> None:
    parent = str(Path(path).parent)
    _run_remote_shell(
        ssh,
        host,
        ssh_args,
        f"mkdir -p {_quote_remote_path(parent)} && cat > {_quote_remote_path(path)} && chmod +x {_quote_remote_path(path)}",
        input_text=text,
    )


def _read_remote_text(ssh: str, host: str, ssh_args: list[str], path: str, *, missing_ok: bool) -> str:
    quoted_path = _quote_remote_path(path)
    command = f"cat {quoted_path}"
    if missing_ok:
        command = f"test -f {quoted_path} && cat {quoted_path} || true"
    return _run_remote_shell(ssh, host, ssh_args, command).stdout or ""


def _quote_remote_path(path: str) -> str:
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def _sync_to_remote(
    rsync: str,
    *,
    source: Path,
    host: str,
    destination: str,
    ssh_args: list[str],
    rsync_args: list[str],
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"dataset output not found: {source}")
    proc = subprocess.run(
        [
            rsync,
            *rsync_args,
            "-e",
            " ".join(["ssh", *[shlex.quote(item) for item in ssh_args]]),
            str(source).rstrip("/") + "/",
            f"{host}:{destination.rstrip('/')}/",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)


def _sync_from_remote(
    rsync: str,
    *,
    host: str,
    source: str,
    destination: Path,
    ssh_args: list[str],
    rsync_args: list[str],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            rsync,
            *rsync_args,
            "-e",
            " ".join(["ssh", *[shlex.quote(item) for item in ssh_args]]),
            f"{host}:{source.rstrip('/')}/",
            str(destination).rstrip("/") + "/",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)


def _write_external_terminal_script(
    *,
    script_path: Path,
    cwd: Path,
    args: tuple[str, ...],
    env: dict[str, str],
    log_path: Path,
    exit_code_path: Path,
    hold_open: bool,
) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env_lines = [
        f"export {key}={shlex.quote(value)}"
        for key, value in sorted(env.items())
        if _should_export_to_terminal(key, value)
    ]
    command_array = " ".join(shlex.quote(item) for item in args)
    hold = "1" if hold_open else "0"
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set +e",
            f"cd {shlex.quote(str(cwd))}",
            *env_lines,
            f"mkdir -p {shlex.quote(str(log_path.parent))}",
            f": > {shlex.quote(str(log_path))}",
            f"cmd=({command_array})",
            "{",
            "  echo \"[RoboLineage] external terminal training started\"",
            f"  echo \"[RoboLineage] cwd={shlex.quote(str(cwd))}\"",
            "  printf '[RoboLineage] command:'",
            "  printf ' %q' \"${cmd[@]}\"",
            "  echo",
            "  \"${cmd[@]}\"",
            "  cmd_status=$?",
            "  echo \"[RoboLineage] command_exit_code=${cmd_status}\"",
            "  exit \"${cmd_status}\"",
            f"}} 2>&1 | tee -a {shlex.quote(str(log_path))}",
            "cmd_status=${PIPESTATUS[0]}",
            f"tmp={shlex.quote(str(exit_code_path))}.tmp",
            "printf '%s\\n' \"${cmd_status}\" > \"${tmp}\"",
            f"mv \"${{tmp}}\" {shlex.quote(str(exit_code_path))}",
            f"if [ {shlex.quote(hold)} = 1 ]; then",
            "  echo",
            "  read -r -p \"RoboLineage training finished. Press Enter to close this terminal...\" _",
            "fi",
            "exit \"${cmd_status}\"",
            "",
        ]
    )
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)


def _should_export_to_terminal(key: str, value: str) -> bool:
    if not key or key[0].isdigit() or not key.replace("_", "").isalnum():
        return False
    return key.startswith("ROBOLINEAGE_") or os.environ.get(key) != value


def _terminal_launcher_args(script_path: Path, execution: FrameworkExecution) -> list[str]:
    context = {"script": str(script_path), "title": "RoboLineage training"}
    if execution.terminal_command:
        rendered = [part.format(**context) for part in execution.terminal_command]
        if not any("{script}" in part for part in execution.terminal_command):
            rendered.append(str(script_path))
        return rendered

    candidates = [
        ("gnome-terminal", ["gnome-terminal", "--", "bash", str(script_path)]),
        ("x-terminal-emulator", ["x-terminal-emulator", "-e", "bash", str(script_path)]),
        ("konsole", ["konsole", "-e", "bash", str(script_path)]),
        ("xfce4-terminal", ["xfce4-terminal", "--command", f"bash {shlex.quote(str(script_path))}"]),
        ("xterm", ["xterm", "-e", "bash", str(script_path)]),
    ]
    for binary, args in candidates:
        if shutil.which(binary):
            return args
    raise FileNotFoundError(
        "no supported terminal emulator found; set execution.terminal_command in the framework profile"
    )


def _write_selected_symlinks(rollouts: list[SelectedRollout], selected_dir: Path) -> None:
    selected_dir.mkdir(parents=True, exist_ok=True)
    for rollout in rollouts:
        target = selected_dir / rollout.rollout_id
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                target.unlink()
            else:
                raise FileExistsError(f"refusing to replace non-symlink staging target: {target}")
        target.symlink_to(Path(rollout.rollout_dir), target_is_directory=True)


def _dataset_adapter_unresolved_reason(profile: FrameworkProfile) -> str | None:
    adapter = dict(profile.dataset_adapter or {})
    target_contract = adapter.get("target_contract") if isinstance(adapter.get("target_contract"), dict) else {}
    candidate = adapter.get("adapter_candidate") if isinstance(adapter.get("adapter_candidate"), dict) else {}
    adapter_id = str(adapter.get("adapter_id") or "")
    candidate_id = str(candidate.get("adapter_id") or "")
    strategy = str(adapter.get("strategy") or "")
    candidate_strategy = str(candidate.get("strategy") or "")
    target_kind = str(
        target_contract.get("dataset_kind")
        or adapter.get("target_dataset_kind")
        or candidate.get("target_dataset_kind")
        or ""
    )
    if adapter_id in {"missing_dataset_converter", "unresolved_custom_dataset"}:
        return "dataset_adapter_unresolved"
    if candidate_id in {"missing_dataset_converter", "unresolved_custom_dataset"}:
        return "dataset_adapter_unresolved"
    if strategy in {"requires_dataset_command", "requires_dataset_adapter"}:
        return "dataset_adapter_unresolved"
    if candidate_strategy in {"requires_dataset_command", "requires_dataset_adapter"}:
        return "dataset_adapter_unresolved"
    if target_kind == "unknown_custom":
        return "dataset_contract_unknown"
    return None


def _dataset_adapt_status_allows_training(status: dict[str, Any]) -> bool:
    status_value = str(status.get("status") or "")
    if status_value == "completed":
        return True
    if status_value != "skipped":
        return False
    adapter_id = str(status.get("adapter_id") or "")
    strategy = str(status.get("adapter_strategy") or "")
    target_kind = str(status.get("target_dataset_kind") or "")
    if target_kind == "unknown_custom":
        return False
    return adapter_id in {"direct_selected_rollouts_file", "no_conversion_required"} and strategy in {
        "direct_manifest",
        "no_conversion_required",
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _write_dataset_adapt_status(
    path: Path,
    *,
    status: str,
    profile: FrameworkProfile,
    context: dict[str, str],
    selected_rollout_count: int,
    adapter_plan_path: Path | None,
    adapter_report: dict[str, Any] | None = None,
    reason: str | None = None,
    error: str | None = None,
) -> None:
    adapter = dict(profile.dataset_adapter or {})
    target_contract = adapter.get("target_contract") if isinstance(adapter.get("target_contract"), dict) else {}
    candidate = adapter.get("adapter_candidate") if isinstance(adapter.get("adapter_candidate"), dict) else {}
    warnings = _dedupe(
        [
            *[str(item) for item in adapter.get("warnings") or []],
            *[str(item) for item in candidate.get("warnings") or []],
            *[str(item) for item in target_contract.get("warnings") or []],
        ]
    )
    payload = {
        "schema_version": "RoboLineage.dataset_adapt_status.v1",
        "status": status,
        "framework": profile.name,
        "framework_type": profile.framework_type,
        "adapter_id": adapter.get("adapter_id") or candidate.get("adapter_id"),
        "adapter_strategy": adapter.get("strategy") or candidate.get("strategy"),
        "target_dataset_kind": target_contract.get("dataset_kind"),
        "dataset_output": context["dataset_output"],
        "selected_rollout_count": selected_rollout_count,
        "adapter_plan_path": str(adapter_plan_path) if adapter_plan_path is not None else None,
        "adapter_report": adapter_report,
        "reason": reason,
        "warnings": warnings,
        "error": error,
        "updated_at": _now_iso(),
    }
    _write_json_atomic(path, payload)


def _find_dataset_adapter_report(dataset_output: Path) -> dict[str, Any] | None:
    for rel in (
        "ROBOLINEAGE_generated_dataset_report.json",
        "export.json",
        "dataset_adapter_report.json",
        "adapter_report.json",
    ):
        report = _read_json(dataset_output / rel)
        if report:
            return {"path": str(dataset_output / rel), **report}
    episodes = sorted(dataset_output.glob("episode_*.hdf5"))
    if episodes:
        return {
            "path": None,
            "exported_episode_count": len(episodes),
            "episodes": [{"episode_path": str(path)} for path in episodes[:20]],
        }
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _coerce_scalar(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _jsonable_asdict(value: Any) -> dict[str, Any]:
    data = asdict(value)
    return {key: _jsonable(item) for key, item in data.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    schema_name = _VALIDATED_ARTIFACT_SCHEMAS.get(path.name)
    if schema_name is not None:
        write_validated_json_atomic(path, data, schema_name)
        return
    _write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _write_running_status(
    path: Path,
    framework: str,
    context: dict[str, str],
    command_name: str,
    log_path: Path,
    metrics: dict[str, Any],
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "running",
            "framework": framework,
            "dataset_version": context["dataset_version"],
            "policy_version": context["policy_version"],
            "current_command": command_name,
            "current_log_path": str(log_path),
            "metrics": metrics,
            "updated_at": _now_iso(),
        },
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
