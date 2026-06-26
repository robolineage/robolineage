from __future__ import annotations

import json
import os
import re
import shlex
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from robolineage_shared_agents.llm_routes import resolve_ai_route

from .dataset_adapter_agent import DatasetAdapterAgent
from .dataset_adapters.registry import (
    choose_adapter_candidate,
    infer_target_data_contract,
    registry_summary_payload,
)


DISCOVERY_SCHEMA_VERSION = "RoboLineage.framework_discovery.v1"
UNDERSTANDING_SCHEMA_VERSION = "RoboLineage.framework_understanding.v1"
INTEGRATION_MANIFEST_SCHEMA_VERSION = "RoboLineage.training_integration_manifest.v1"
GENERATED_ADAPTER_SCHEMA_VERSION = "RoboLineage.generated_dataset_adapter.v1"
GENERATED_MONITOR_SCHEMA_VERSION = "RoboLineage.generated_training_monitor.v1"


@dataclass(frozen=True)
class FrameworkDiscoveryResult:
    name: str
    framework_type: str
    repo_root: Path
    output_dir: Path
    profile_path: Path
    discovery_path: Path
    report_path: Path
    understanding_path: Path | None = None
    understanding_report_path: Path | None = None
    events_path: Path | None = None
    adapter_registry_path: Path | None = None
    target_contract_path: Path | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


@dataclass(frozen=True)
class CommandIntake:
    dataset_command: str | tuple[str, ...] | None = None
    train_command: str | tuple[str, ...] | None = None
    eval_command: str | tuple[str, ...] | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedFrameworkArtifacts:
    dataset_command: tuple[str, ...] | None = None
    train_command: tuple[str, ...] | None = None
    dataset_adapter: dict[str, Any] | None = None
    monitor: dict[str, Any] | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    log_patterns: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class FrameworkDiscoveryLLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class OpenAICompatibleDiscoveryClient:
    """Minimal OpenAI-compatible chat client for optional repo understanding."""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> "OpenAICompatibleDiscoveryClient | None":
        route = resolve_ai_route(
            "ROBOLINEAGE_DISCOVERY_LLM",
            fallback_prefixes=("ROBOLINEAGE_AGENT", "TASK_LLM", "OPENAI"),
            base_url_default="https://api.openai.com/v1",
            timeout_default=60.0,
        )
        api_key = route.api_key
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=route.model,
            base_url=(route.base_url or "https://api.openai.com/v1").rstrip("/"),
            timeout_s=float(route.timeout_s or 60.0),
        )

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are RoboLineage Framework Discovery Agent. Understand an unknown robot policy "
                        "training repository from commands, file tree and snippets. Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"])


class FrameworkDiscoveryAgent:
    """Turns a user's existing training commands into a reusable RoboLineage profile.

    The user tells RoboLineage how they normally run dataset build / train / eval.
    Discovery records the surrounding repository context, asks
    DatasetAdapterAgent for a dataset conversion plan, and writes a
    deterministic profile for later lifecycle runs.
    """

    def __init__(self, *, llm_client: FrameworkDiscoveryLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def discover(
        self,
        *,
        repo_root: Path,
        output_dir: Path,
        commands: CommandIntake,
        target_dataset_format: str | None = None,
        command_context: str | None = None,
        name: str | None = None,
        framework_type: str | None = None,
        fixed_input_dir: str | None = None,
        checkpoint_glob: str | None = None,
        train_log: str | None = None,
        eval_result: str | None = None,
        log_patterns: dict[str, str] | None = None,
        train_launch_mode: str = "inline",
        terminal_command: str | tuple[str, ...] | None = None,
        terminal_hold_open: bool = True,
        enable_llm_understanding: bool = False,
    ) -> FrameworkDiscoveryResult:
        repo = Path(repo_root).expanduser().resolve()
        if not repo.exists() or not repo.is_dir():
            raise FileNotFoundError(f"training repository not found: {repo}")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        events = _DiscoveryEventLog(out / "framework_discovery_events.jsonl")
        events.emit("discovery_started", repo_root=str(repo), name=name or repo.name)

        repo_files = _scan_repo(repo)
        inferred_type = framework_type or _infer_framework_type(repo, repo_files, commands)
        profile_name = _safe_name(name or repo.name or "training_framework")
        repo_summary = _repo_summary(repo_files)
        events.emit(
            "repo_scanned",
            file_count=len(repo_files),
            framework_type=inferred_type,
            important_files=repo_summary.get("important_files") or [],
        )
        target_dataset_format = str(target_dataset_format or "").strip()
        command_context = str(command_context or "").strip()
        target_contract = infer_target_data_contract(
            repo_root=repo,
            repo_files=repo_files,
            target_dataset_format=target_dataset_format,
            command_context=command_context,
            framework_type=inferred_type,
        )
        adapter_candidate = choose_adapter_candidate(target_contract)
        registry_payload = registry_summary_payload(
            contract=target_contract,
            candidate=adapter_candidate,
        )
        target_contract_path = out / "target_data_contract.json"
        adapter_registry_path = out / "dataset_adapter_registry.json"
        _write_json_atomic(target_contract_path, target_contract.to_dict())
        _write_json_atomic(adapter_registry_path, registry_payload)
        events.emit(
            "target_contract_inferred",
            dataset_kind=target_contract.dataset_kind,
            camera_names=list(target_contract.camera_names),
            confidence=target_contract.confidence,
            warnings=list(target_contract.warnings),
        )
        events.emit(
            "adapter_candidate_selected",
            adapter_id=adapter_candidate.adapter_id,
            strategy=adapter_candidate.strategy,
            confidence=adapter_candidate.confidence,
        )
        generated = _generate_framework_artifacts(
            repo=repo,
            repo_files=repo_files,
            output_dir=out,
            target_dataset_format=target_dataset_format,
            command_context=command_context,
            commands=commands,
            framework_type=inferred_type,
            target_contract=target_contract.to_dict(),
            adapter_candidate=adapter_candidate.to_dict(),
        )
        discovery_commands = commands
        if generated.dataset_command is not None or generated.train_command is not None:
            discovery_commands = CommandIntake(
                dataset_command=generated.dataset_command or commands.dataset_command,
                train_command=generated.train_command or commands.train_command,
                eval_command=commands.eval_command,
                env=commands.env,
            )
        dataset_adapter = DatasetAdapterAgent().plan(
            repo_root=repo,
            framework_type=inferred_type,
            dataset_command=discovery_commands.dataset_command,
            train_command=discovery_commands.train_command,
        )
        if generated.dataset_adapter is not None:
            dataset_adapter_payload = {
                **generated.dataset_adapter,
                "target_contract": target_contract.to_dict(),
                "adapter_candidate": adapter_candidate.to_dict(),
            }
        else:
            planned_adapter_payload = dataset_adapter.to_dict()
            if (
                adapter_candidate.strategy in {"requires_dataset_adapter", "requires_dataset_command"}
                and planned_adapter_payload.get("adapter_id") == "no_conversion_required"
            ):
                planned_adapter_payload = {
                    **planned_adapter_payload,
                    "adapter_id": adapter_candidate.adapter_id,
                    "strategy": adapter_candidate.strategy,
                    "confidence": adapter_candidate.confidence,
                    "warnings": _dedupe(
                        [
                            *[str(item) for item in planned_adapter_payload.get("warnings") or []],
                            *list(adapter_candidate.warnings),
                        ]
                    ),
                }
            dataset_adapter_payload = {
                **planned_adapter_payload,
                "target_contract": target_contract.to_dict(),
                "adapter_candidate": adapter_candidate.to_dict(),
            }
        effective_commands = discovery_commands
        if effective_commands.dataset_command is None and dataset_adapter.dataset_command is not None:
            effective_commands = CommandIntake(
                dataset_command=dataset_adapter.dataset_command,
                train_command=effective_commands.train_command,
                eval_command=effective_commands.eval_command,
                env=effective_commands.env,
            )
        events.emit("deep_discovery_started", mode="deep_repo_understanding")
        deep_inspection = _deep_inspect_training_repo(
            repo=repo,
            repo_files=repo_files,
            commands=effective_commands,
            target_dataset_format=target_dataset_format,
            command_context=command_context,
            events=events,
        )
        warnings = tuple(
            _dedupe(
                [
                    *_warnings(repo_files, effective_commands, fixed_input_dir),
                    *dataset_adapter.warnings,
                    *generated.warnings,
                ]
            )
        )
        deep_outputs = deep_inspection.get("outputs") if isinstance(deep_inspection.get("outputs"), dict) else {}
        effective_outputs = {
            "checkpoint_glob": _preferred_checkpoint_glob(
                user_value=checkpoint_glob,
                generated_value=generated.outputs.get("checkpoint_glob"),
                deep_candidates=deep_outputs.get("checkpoint_glob_candidates"),
            ),
            "train_log": train_log or generated.outputs.get("train_log") or _first_string(deep_outputs.get("train_log_candidates")),
            "eval_result": eval_result or generated.outputs.get("eval_result"),
        }
        effective_log_patterns = {**generated.log_patterns, **(log_patterns or {})}
        profile = _profile_payload(
            name=profile_name,
            framework_type=inferred_type,
            repo_root=repo,
            commands=effective_commands,
            fixed_input_dir=fixed_input_dir,
            checkpoint_glob=effective_outputs["checkpoint_glob"],
            train_log=effective_outputs["train_log"],
            eval_result=effective_outputs["eval_result"],
            log_patterns=effective_log_patterns,
            dataset_adapter=dataset_adapter_payload,
            monitor=generated.monitor or {},
            train_launch_mode=train_launch_mode,
            terminal_command=terminal_command,
            terminal_hold_open=terminal_hold_open,
        )
        understanding: dict[str, Any] | None = None
        understanding_path: Path | None = None
        understanding_report_path: Path | None = None
        if enable_llm_understanding:
            events.emit("llm_understanding_started", enabled=True)
            understanding = _understand_framework(
                self,
                repo=repo,
                repo_files=repo_files,
                repo_summary=repo_summary,
                commands=effective_commands,
                target_dataset_format=target_dataset_format,
                command_context=command_context,
                initial_profile=profile,
                deep_inspection=deep_inspection,
            )
            _apply_understanding_to_profile(
                profile,
                understanding,
                user_supplied_framework_type=framework_type is not None,
                user_supplied_outputs={
                    "checkpoint_glob": checkpoint_glob is not None or "checkpoint_glob" in generated.outputs,
                    "train_log": train_log is not None or "train_log" in generated.outputs,
                    "eval_result": eval_result is not None or "eval_result" in generated.outputs,
                },
                user_supplied_fixed_input_dir=fixed_input_dir is not None,
            )
            inferred_type = str(profile.get("framework_type") or inferred_type)
            warnings = tuple(
                _dedupe(
                    [
                        *warnings,
                        *[str(item) for item in understanding.get("warnings") or []],
                    ]
                )
            )
            understanding_path = out / "framework_understanding.json"
            understanding_report_path = out / "framework_understanding_report.md"
            _write_json_atomic(understanding_path, understanding)
            _write_text_atomic(understanding_report_path, _render_understanding_report(understanding))
            events.emit(
                "llm_understanding_completed",
                status=understanding.get("status"),
                confidence=understanding.get("confidence"),
                warnings=understanding.get("warnings") or [],
            )

        discovery = {
            "schema_version": DISCOVERY_SCHEMA_VERSION,
            "name": profile_name,
            "framework_type": inferred_type,
            "repo_root": str(repo),
            "created_at": _now_iso(),
            "agent_intake": {
                "target_dataset_format": target_dataset_format,
                "command_context": command_context,
            },
            "command_intake": {
                "dataset_command": _command_parts(effective_commands.dataset_command),
                "train_command": _command_parts(effective_commands.train_command),
                "eval_command": _command_parts(effective_commands.eval_command),
                "env": effective_commands.env,
            },
            "staging": profile["staging"],
            "outputs": profile["outputs"],
            "execution": profile.get("execution") or {},
            "dataset_adapter": profile.get("dataset_adapter") or {},
            "target_contract": target_contract.to_dict(),
            "adapter_registry": registry_payload,
            "monitor": profile.get("monitor") or {},
            "log_patterns": profile.get("log_patterns") or {},
            "deep_inspection": deep_inspection,
            "repo_summary": repo_summary,
            "warnings": list(warnings),
        }
        if understanding is not None:
            discovery["llm_understanding"] = _understanding_summary(
                understanding,
                understanding_path=understanding_path,
                understanding_report_path=understanding_report_path,
            )

        profile_path = out / "framework_profile.generated.yaml"
        discovery_path = out / "framework_discovery.json"
        report_path = out / "framework_discovery_report.md"
        integration_manifest_path = out / "training_integration_manifest.json"
        integration_manifest = _training_integration_manifest(
            profile=profile,
            discovery=discovery,
            deep_inspection=deep_inspection,
            understanding=understanding,
        )
        _write_text_atomic(profile_path, yaml.safe_dump(profile, sort_keys=False, allow_unicode=True))
        _write_json_atomic(integration_manifest_path, integration_manifest)
        events.emit(
            "profile_written",
            profile_path=str(profile_path),
            dataset_command=bool(profile.get("dataset_command")),
            train_command=bool(profile.get("train_command")),
            monitor=bool(profile.get("monitor")),
        )
        events.emit("integration_manifest_written", manifest_path=str(integration_manifest_path))
        events.emit("discovery_completed", profile_path=str(profile_path), warnings=list(warnings))
        discovery["events"] = events.read_events()
        discovery["events_path"] = str(events.path)
        discovery["target_contract_path"] = str(target_contract_path)
        discovery["adapter_registry_path"] = str(adapter_registry_path)
        discovery["integration_manifest"] = integration_manifest
        discovery["integration_manifest_path"] = str(integration_manifest_path)
        _write_json_atomic(discovery_path, discovery)
        _write_text_atomic(report_path, _render_report(discovery))
        return FrameworkDiscoveryResult(
            name=profile_name,
            framework_type=inferred_type,
            repo_root=repo,
            output_dir=out,
            profile_path=profile_path,
            discovery_path=discovery_path,
            report_path=report_path,
            understanding_path=understanding_path,
            understanding_report_path=understanding_report_path,
            events_path=events.path,
            adapter_registry_path=adapter_registry_path,
            target_contract_path=target_contract_path,
            warnings=warnings,
        )

def _understand_framework(
        self,
        *,
        repo: Path,
        repo_files: list[str],
        repo_summary: dict[str, Any],
        commands: CommandIntake,
        target_dataset_format: str,
        command_context: str,
        initial_profile: dict[str, Any],
        deep_inspection: dict[str, Any],
    ) -> dict[str, Any]:
        client = self.llm_client or OpenAICompatibleDiscoveryClient.from_env()
        snippets = _collect_repo_snippets(repo, repo_files, commands, max_files=60, max_chars_per_file=8000)
        base: dict[str, Any] = {
            "schema_version": UNDERSTANDING_SCHEMA_VERSION,
            "created_at": _now_iso(),
            "used_llm": client is not None,
            "status": "skipped" if client is None else "running",
            "model": getattr(client, "model", None) if client is not None else None,
            "repo_root": str(repo),
            "snippet_files": [item["path"] for item in snippets],
        }
        if client is None:
            return {
                **base,
                "status": "skipped",
                "reason": "llm_client_or_api_key_missing",
                "confidence": 0.0,
                "profile_patch": {},
                "assumptions": ["LLM understanding was requested but no discovery LLM client/API key is configured."],
                "warnings": ["llm_understanding_skipped"],
            }

        prompt = _understanding_prompt(
            repo=repo,
            repo_summary=repo_summary,
            commands=commands,
            target_dataset_format=target_dataset_format,
            command_context=command_context,
            initial_profile=initial_profile,
            deep_inspection=deep_inspection,
            snippets=snippets,
        )
        try:
            raw = client.complete(prompt)
        except Exception as exc:
            return {
                **base,
                "status": "failed",
                "reason": str(exc),
                "confidence": 0.0,
                "profile_patch": {},
                "assumptions": ["LLM understanding failed; deterministic discovery profile is still usable."],
                "warnings": ["llm_understanding_failed"],
            }

        parsed = _parse_json_object(raw) or {}
        normalized = _normalize_understanding(parsed)
        return {
            **base,
            **normalized,
            "status": "completed" if parsed else "failed",
            "raw_response": raw,
        }


class _DiscoveryEventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, **payload: Any) -> None:
        record = {
            "event": event,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not self.path.exists():
            return events
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
        return events


def _generate_framework_artifacts(
    *,
    repo: Path,
    repo_files: list[str],
    output_dir: Path,
    target_dataset_format: str,
    command_context: str,
    commands: CommandIntake,
    framework_type: str,
    target_contract: dict[str, Any] | None = None,
    adapter_candidate: dict[str, Any] | None = None,
) -> GeneratedFrameworkArtifacts:
    monitor = _generate_monitor_spec(
        repo=repo,
        repo_files=repo_files,
        output_dir=output_dir,
    )
    contract_kind = str((target_contract or {}).get("dataset_kind") or "")
    if contract_kind and contract_kind != "act_hdf5":
        return GeneratedFrameworkArtifacts(monitor=monitor or None)

    if contract_kind != "act_hdf5" and not _needs_generated_act_hdf5_adapter(
        repo=repo,
        repo_files=repo_files,
        target_dataset_format=target_dataset_format,
    ):
        return GeneratedFrameworkArtifacts(monitor=monitor or None)

    camera_names = _target_camera_names(target_dataset_format, command_context)
    adapter_plan_path = output_dir / "dataset_adapter_plan.json"
    adapter_payload = {
        "schema_version": GENERATED_ADAPTER_SCHEMA_VERSION,
        "adapter_id": "rosbag_act_hdf5",
        "strategy": "registered_adapter_module",
        "confidence": 0.86,
        "module": "robolineage_train.dataset_adapters.rosbag_act_hdf5",
        "plan_path": str(adapter_plan_path),
        "target_contract": target_contract or {},
        "adapter_candidate": adapter_candidate or {},
        "target_format": "ACT episode_*.hdf5",
        "dataset_command": [
            sys.executable,
            "-m",
            "robolineage_train.dataset_adapters.rosbag_act_hdf5",
            "--selected-rollouts",
            "{selected_rollouts_file}",
            "--output-dir",
            "{dataset_output}",
            "--overwrite",
            "--target-hz",
            "30",
            "--camera-names",
            *camera_names,
        ],
        "train_input": "{dataset_output}",
        "output_path": "{dataset_output}",
        "source_data_policy": "read_only",
        "generated_files": [],
        "assumptions": [
            "The target training repository reads contiguous episode_*.hdf5 files from DATASET_DIR or --datasets.",
            "RoboLineage raw rollouts contain raw/rosbag2 captured directly from ROS2 topics.",
            "The registered adapter resamples episodes onto a fixed 30 Hz training timeline.",
            "The registered adapter maps robot-state vectors into qpos/qvel/effort/eef/action fields.",
        ],
        "warnings": [
            "Validate a sample rosbag conversion before long training runs."
        ],
        "created_at": _now_iso(),
    }
    _write_json_atomic(adapter_plan_path, adapter_payload)

    train_command = _wrap_train_command_for_generated_dataset(
        commands.train_command,
        camera_names=tuple(camera_names),
    )
    outputs = {
        "checkpoint_glob": _infer_checkpoint_glob(repo, repo_files, command_context),
        "train_log": "{workspace_dir}/train_command.log",
    }
    log_patterns = _infer_generated_log_patterns(repo, repo_files)
    warnings: list[str] = []
    if train_command is None and commands.train_command is not None:
        warnings.append("generated_adapter_train_command_not_wrapped")
    return GeneratedFrameworkArtifacts(
        dataset_command=tuple(adapter_payload["dataset_command"]),
        train_command=train_command,
        dataset_adapter=adapter_payload,
        monitor=monitor or None,
        outputs=outputs,
        log_patterns=log_patterns,
        warnings=tuple(warnings),
    )


def _needs_generated_act_hdf5_adapter(
    *,
    repo: Path,
    repo_files: list[str],
    target_dataset_format: str,
) -> bool:
    target = target_dataset_format.lower()
    if "hdf5" not in target and ".hdf5" not in target:
        return False
    if any(token in target for token in ("episode_", "observations/qpos", "/action", "action_eef")):
        return True
    snippets = _read_repo_text(repo, repo_files, max_chars=80000).lower()
    return "episode_" in snippets and "observations/qpos" in snippets and "h5py" in snippets


def _target_camera_names(target_dataset_format: str, command_context: str) -> list[str]:
    explicit = _camera_names_from_command_context(command_context)
    if explicit:
        return explicit
    text = f"{target_dataset_format}\n{command_context}".lower()
    known = ("head", "left_wrist", "right_wrist", "camera_h", "camera_l", "camera_r")
    found = [name for name in known if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)]
    if not found:
        return ["head", "right_wrist"]
    # Prefer semantic camera names in generated ACT HDF5 when both aliases appear.
    aliases = {"camera_h": "head", "camera_l": "left_wrist", "camera_r": "right_wrist"}
    normalized: list[str] = []
    for name in found:
        value = aliases.get(name, name)
        if value not in normalized:
            normalized.append(value)
    return normalized


def _camera_names_from_command_context(command_context: str) -> list[str]:
    aliases = {"camera_h": "head", "camera_l": "left_wrist", "camera_r": "right_wrist"}
    patterns = (
        r"CAMERA_NAMES\s*=\s*['\"]([^'\"]+)['\"]",
        r"CAMERA_NAMES\s*=\s*([A-Za-z0-9_, -]+)",
        r"--camera[-_]names?\s+([A-Za-z0-9_, -]+)",
    )
    known = {"head", "left_wrist", "right_wrist", "camera_h", "camera_l", "camera_r"}
    for pattern in patterns:
        match = re.search(pattern, command_context, re.IGNORECASE)
        if not match:
            continue
        tokens = [
            token.strip()
            for token in re.split(r"[,\s]+", match.group(1))
            if token.strip()
        ]
        out: list[str] = []
        for token in tokens:
            value = aliases.get(token.lower(), token.lower())
            if token.lower() in known and value not in out:
                out.append(value)
        if out:
            return out
    return []


def _wrap_train_command_for_generated_dataset(
    command: str | tuple[str, ...] | None,
    *,
    camera_names: tuple[str, ...] = (),
) -> tuple[str, ...] | None:
    parts = _command_parts(command)
    if not parts:
        return None
    joined = " ".join(parts)
    if "{dataset_output}" in joined or "{checkpoint_dir}" in joined:
        return parts
    shell_command = " ".join(shlex.quote(part) for part in parts)
    env_assignments = [
        "DISPLAY=",
        "WAYLAND_DISPLAY=",
        "DATASET_DIR={dataset_output}",
        "CKPT_DIR={checkpoint_dir}",
    ]
    if camera_names:
        env_assignments.append("CAMERA_NAMES=" + shlex.quote(" ".join(camera_names)))
    return (
        "bash",
        "-lc",
        " ".join(env_assignments) + " " + shell_command,
    )


def _infer_checkpoint_glob(repo: Path, repo_files: list[str], command_context: str) -> str:
    context_match = re.search(r"(?:checkpoint output|checkpoint_glob|checkpoint glob)\s*[:=]\s*(.+)", command_context, re.IGNORECASE)
    if context_match:
        value = context_match.group(1).strip()
        if value:
            return value
    text = _read_repo_text(repo, repo_files, max_chars=120000)
    if "policy_best.ckpt" in text:
        return "{checkpoint_dir}/policy_best.ckpt"
    if ".ckpt" in text:
        return "{checkpoint_dir}/*.ckpt"
    if ".pth" in text or ".pt" in text:
        return "{checkpoint_dir}/*.{pt,pth}"
    return "{checkpoint_dir}/*"


def _infer_generated_log_patterns(repo: Path, repo_files: list[str]) -> dict[str, str]:
    text = _read_repo_text(repo, repo_files, max_chars=120000)
    patterns: dict[str, str] = {}
    if "best_val_loss" in text:
        patterns["best_val_loss"] = r'"best_val_loss"\s*:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)'
    if "val_loss" in text:
        patterns["val_loss"] = r'"val_loss"\s*:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)'
    return patterns


def _generate_monitor_spec(
    *,
    repo: Path,
    repo_files: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    text = _read_repo_text(repo, repo_files, max_chars=160000)
    prefixes = _extract_json_log_prefixes(text)
    events = sorted(set(re.findall(r"emit_web_progress\(\s*[\"']([^\"']+)[\"']", text)))
    if not prefixes and not events:
        return {}
    spec_path = output_dir / "generated_training_monitor.json"
    spec = {
        "schema_version": GENERATED_MONITOR_SCHEMA_VERSION,
        "strategy": "generated_from_repo_training_log_code",
        "json_line_prefixes": prefixes or [],
        "event_field": "event",
        "events": events,
        "source": "repo_scan",
        "spec_path": str(spec_path),
        "created_at": _now_iso(),
    }
    _write_json_atomic(spec_path, spec)
    return spec


def _extract_json_log_prefixes(text: str) -> list[str]:
    prefixes: list[str] = []
    for pattern in (
        r"WEB_PROGRESS_PREFIX\s*=\s*[\"']([^\"']+)[\"']",
        r"PROGRESS_PREFIX\s*=\s*[\"']([^\"']+)[\"']",
        r"JSON_LOG_PREFIX\s*=\s*[\"']([^\"']+)[\"']",
    ):
        for value in re.findall(pattern, text):
            if value not in prefixes:
                prefixes.append(value)
    return prefixes


def _read_repo_text(repo: Path, repo_files: list[str], *, max_chars: int) -> str:
    parts: list[str] = []
    remaining = max_chars
    for rel in repo_files:
        if remaining <= 0:
            break
        lower = rel.lower()
        if not any(token in lower for token in ("train", "dataset", "convert", "collect", "utils", "readme", "02_train")):
            continue
        path = repo / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text:
            continue
        chunk = text[:remaining]
        parts.append(f"\n# file: {rel}\n{chunk}")
        remaining -= len(chunk)
    return "\n".join(parts)




def _profile_payload(
    *,
    name: str,
    framework_type: str,
    repo_root: Path,
    commands: CommandIntake,
    fixed_input_dir: str | None,
    checkpoint_glob: str | None,
    train_log: str | None,
    eval_result: str | None,
    log_patterns: dict[str, str],
    dataset_adapter: dict[str, Any],
    monitor: dict[str, Any],
    train_launch_mode: str,
    terminal_command: str | tuple[str, ...] | None,
    terminal_hold_open: bool,
) -> dict[str, Any]:
    staging: dict[str, str] = {
        "selected_rollouts_file": "{staging_dir}/selected_rollouts.json",
    }
    if _commands_need_rollout_dir(commands) or fixed_input_dir:
        staging["selected_rollouts_dir"] = "{staging_dir}/selected_rollouts"
    if fixed_input_dir:
        staging["framework_input_dir"] = fixed_input_dir
    profile: dict[str, Any] = {
        "schema_version": "RoboLineage.framework_profile.v1",
        "name": name,
        "framework_type": framework_type,
        "adapter_version": "0.2",
        "repo_root": str(repo_root),
        "staging": staging,
        "outputs": {
            "checkpoint_glob": checkpoint_glob or "{checkpoint_dir}/*",
            "train_log": train_log or "{checkpoint_dir}/training.log",
            "eval_result": eval_result or "{eval_output}/result.json",
        },
        "dataset_adapter": dataset_adapter,
        "monitor": monitor,
        "log_patterns": log_patterns,
    }
    if commands.dataset_command is not None:
        profile["dataset_command"] = _command_payload(commands.dataset_command, commands.env)
    if commands.train_command is not None:
        profile["train_command"] = _command_payload(commands.train_command, commands.env)
    if commands.eval_command is not None:
        profile["eval_command"] = _command_payload(commands.eval_command, commands.env)
    if train_launch_mode != "inline" or terminal_command is not None or terminal_hold_open is not True:
        profile["execution"] = {
            "train_launch_mode": train_launch_mode,
            "terminal_hold_open": bool(terminal_hold_open),
        }
        if terminal_command is not None:
            profile["execution"]["terminal_command"] = list(_command_parts(terminal_command))
    return profile


def _command_payload(command: str | tuple[str, ...], env: dict[str, str]) -> dict[str, Any]:
    return {
        "cwd": "{repo_root}",
        "args": list(_command_parts(command)),
        "env": dict(env),
    }


def _command_parts(command: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if command is None:
        return ()
    if isinstance(command, str):
        return tuple(shlex.split(command))
    return tuple(str(part) for part in command)


def _scan_repo(root: Path, max_files: int = 240) -> list[str]:
    ignored_dirs = {
        ".git",
        ".agent",
        ".venv",
        "__pycache__",
        "node_modules",
        "wandb",
        "runs",
        "checkpoints",
        "outputs",
        "data",
        "datasets",
    }
    interesting_suffixes = {".py", ".sh", ".yaml", ".yml", ".toml", ".json", ".md", ".txt"}
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in ignored_dirs or part.startswith(".mypy") for part in rel_parts):
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() not in interesting_suffixes and path.name.lower() not in {"makefile"}:
            continue
        rows.append(str(path.relative_to(root)))
        if len(rows) >= max_files:
            break
    return rows


def _infer_framework_type(repo: Path, files: list[str], commands: CommandIntake) -> str:
    text = " ".join([repo.name, *files, *_command_parts(commands.dataset_command), *_command_parts(commands.train_command), *_command_parts(commands.eval_command)]).lower()
    if "openvla" in text or re.search(r"\bvla\b", text):
        return "vla_like"
    if "diffusion" in text or "dp3" in text:
        return "diffusion_policy_like"
    if "lerobot" in text:
        return "lerobot_like"
    return "generic_policy"


def _warnings(files: list[str], commands: CommandIntake, fixed_input_dir: str | None) -> tuple[str, ...]:
    out: list[str] = []
    joined_commands = " ".join(
        [
            *_command_parts(commands.dataset_command),
            *_command_parts(commands.train_command),
            *_command_parts(commands.eval_command),
        ]
    )
    if commands.dataset_command is None and "{dataset_output}" in joined_commands:
        out.append("dataset_command_missing")
    if commands.train_command is None:
        out.append("train_command_missing")
    if commands.eval_command is None:
        out.append("eval_command_missing")
    if not any(
        token in joined_commands
        for token in (
            "{framework_input_dir}",
            "{selected_rollouts_dir}",
            "{selected_rollouts_file}",
            "{dataset_output}",
        )
    ) and fixed_input_dir is None:
        out.append("commands_do_not_reference_ROBOLINEAGE_inputs")
    if not any(Path(item).name.lower().startswith("readme") for item in files):
        out.append("repo_readme_not_found")
    return tuple(out)


def _commands_need_rollout_dir(commands: CommandIntake) -> bool:
    joined_commands = " ".join(
        [
            *_command_parts(commands.dataset_command),
            *_command_parts(commands.train_command),
            *_command_parts(commands.eval_command),
        ]
    )
    return "{framework_input_dir}" in joined_commands or "{selected_rollouts_dir}" in joined_commands


def _repo_summary(files: list[str]) -> dict[str, Any]:
    return {
        "file_count_scanned": len(files),
        "candidate_dataset_scripts": _candidates(files, ("dataset", "convert", "preprocess", "export")),
        "candidate_train_scripts": _candidates(files, ("train", "finetune", "policy")),
        "candidate_eval_scripts": _candidates(files, ("eval", "evaluate", "rollout", "test")),
        "config_files": [item for item in files if Path(item).suffix.lower() in {".yaml", ".yml", ".json", ".toml"}][:40],
        "files": files[:120],
    }


def _candidates(files: list[str], keywords: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for item in files:
        lower = item.lower()
        if any(keyword in lower for keyword in keywords):
            out.append(item)
    return out[:30]


def _deep_inspect_training_repo(
    *,
    repo: Path,
    repo_files: list[str],
    commands: CommandIntake,
    target_dataset_format: str,
    command_context: str,
    events: _DiscoveryEventLog,
) -> dict[str, Any]:
    command_files = _resolve_command_files(repo, commands)
    expanded_files = _expand_script_references(repo, command_files)
    selected = _dedupe(
        [
            *command_files,
            *expanded_files,
            *_candidates(repo_files, ("train", "finetune", "imitate", "policy")),
            *_candidates(repo_files, ("dataset", "dataloader", "loader", "hdf5", "h5py")),
            *_candidates(repo_files, ("config", "yaml", "params")),
            *_files_with_content_markers(
                repo,
                repo_files,
                (
                    "SummaryWriter",
                    "tensorboard",
                    "tensorboardX",
                    "wandb.init",
                    "torch.save",
                    "save_checkpoint",
                    "policy_best",
                    "DataLoader",
                    "h5py.File",
                ),
                limit=40,
            ),
        ]
    )[:80]

    files_read: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    tensorboard_candidates: list[str] = []
    train_log_candidates: list[str] = []
    checkpoint_candidates: list[str] = []
    dataset_evidence: list[dict[str, Any]] = []
    metrics_sources: set[str] = set()

    for rel in selected:
        path = repo / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        highlights = _extract_deep_highlights(text)
        if not highlights:
            highlights = [{"category": "file_read", "line": 1, "text": "selected as entrypoint/config candidate"}]
        files_read.append(
            {
                "path": rel,
                "chars": len(text),
                "truncated": len(text) > 24000,
                "highlights": highlights[:12],
            }
        )
        for item in highlights:
            category = str(item.get("category") or "")
            line_text = str(item.get("text") or "")
            if category == "tensorboard":
                metrics_sources.add("tensorboard")
                tensorboard_candidates.extend(_extract_logdir_candidates(line_text))
            elif category == "wandb":
                metrics_sources.add("wandb")
            elif category == "checkpoint":
                checkpoint_candidates.extend(_extract_checkpoint_candidates(line_text))
            elif category == "dataset":
                dataset_evidence.append({"path": rel, **item})
        findings.extend({"path": rel, **item} for item in highlights[:12])
        if len(files_read) <= 18:
            events.emit(
                "deep_file_inspected",
                path=rel,
                highlights=[item.get("category") for item in highlights[:6]],
            )

    if not metrics_sources:
        metrics_sources.add("stdout")
    outputs = {
        "stdout_capture": "{workspace_dir}/train_command.log",
        "tensorboard_logdir_candidates": _dedupe(tensorboard_candidates),
        "train_log_candidates": _dedupe(train_log_candidates),
        "checkpoint_glob_candidates": _dedupe(checkpoint_candidates),
        "metrics_sources": sorted(metrics_sources),
    }
    events.emit(
        "deep_outputs_inferred",
        metrics_sources=outputs["metrics_sources"],
        tensorboard_logdir_candidates=outputs["tensorboard_logdir_candidates"][:5],
        checkpoint_glob_candidates=outputs["checkpoint_glob_candidates"][:5],
    )
    return {
        "schema_version": "RoboLineage.deep_training_repo_inspection.v1",
        "mode": "deep",
        "repo_root": str(repo),
        "command_entrypoint_files": command_files,
        "expanded_entrypoint_files": expanded_files,
        "files_read": files_read,
        "findings": findings[:200],
        "outputs": outputs,
        "dataset_evidence": dataset_evidence[:60],
        "user_dataset_description_chars": len(target_dataset_format),
        "user_command_context_chars": len(command_context),
    }


def _resolve_command_files(repo: Path, commands: CommandIntake) -> list[str]:
    out: list[str] = []
    for part in [
        *_command_parts(commands.dataset_command),
        *_command_parts(commands.train_command),
        *_command_parts(commands.eval_command),
    ]:
        if part.startswith("-") or part.startswith("{"):
            continue
        path = Path(part)
        candidates: list[Path] = []
        if path.is_absolute():
            stripped = Path(str(path).lstrip("/"))
            if str(stripped) != ".":
                candidates.append(stripped)
            try:
                candidates.append(path.relative_to(repo))
            except ValueError:
                candidates.append(Path(path.name))
        else:
            candidates.append(path)
        for candidate in candidates:
            if (repo / candidate).is_file():
                out.append(str(candidate))
    return _dedupe(out)


def _expand_script_references(repo: Path, command_files: list[str]) -> list[str]:
    out: list[str] = []
    patterns = (
        r"(?:python|python3|bash|sh)\s+([A-Za-z0-9_./-]+\.(?:py|sh))",
        r"(?:source|\.)\s+([A-Za-z0-9_./-]+\.sh)",
        r"\$\{SCRIPT_DIR\}/([A-Za-z0-9_./-]+\.(?:py|sh))",
    )
    for rel in command_files:
        path = repo / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        base = Path(rel).parent
        for pattern in patterns:
            for match in re.findall(pattern, text):
                candidate = Path(match)
                options = [candidate] if not candidate.is_absolute() else []
                options.append(base / candidate)
                for option in options:
                    normalized = Path(os.path.normpath(str(option)))
                    if (repo / normalized).is_file():
                        out.append(str(normalized))
    return _dedupe(out)


def _files_with_content_markers(repo: Path, repo_files: list[str], markers: tuple[str, ...], *, limit: int) -> list[str]:
    out: list[str] = []
    for rel in repo_files:
        path = repo / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:60000]
        except OSError:
            continue
        if any(marker in text for marker in markers):
            out.append(rel)
            if len(out) >= limit:
                break
    return out


def _extract_deep_highlights(text: str) -> list[dict[str, Any]]:
    categories = {
        "tensorboard": ("SummaryWriter", "tensorboard", "tensorboardX"),
        "wandb": ("wandb.init", "wandb.log"),
        "checkpoint": ("torch.save", "save_checkpoint", "policy_best", "checkpoint", ".ckpt", ".pth"),
        "dataset": ("Dataset", "DataLoader", "h5py.File", "episode_", "qpos", "action_eef", "dataset_dir"),
        "config": ("argparse", "ArgumentParser", "hydra", "yaml.safe_load", "OmegaConf"),
        "metrics": ("loss", "epoch", "step", "Summary", "scalar"),
    }
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for category, markers in categories.items():
            if any(marker in stripped for marker in markers):
                rows.append({"category": category, "line": line_no, "text": stripped[:500]})
                break
        if len(rows) >= 80:
            break
    return rows


def _extract_logdir_candidates(text: str) -> list[str]:
    out: list[str] = []
    for pattern in (
        r"log_dir\s*=\s*([^,\)]+)",
        r"SummaryWriter\(([^,\)]*)",
        r"--(?:logdir|log-dir|log_dir)\s+([^\s]+)",
        r"(?:tensorboard_dir|tb_dir|logdir)\s*=\s*([^\s,\)]+)",
    ):
        for value in re.findall(pattern, text):
            cleaned = str(value).strip().strip("'\"")
            if cleaned and cleaned not in {"None", "True", "False"}:
                out.append(cleaned)
    return out


def _extract_checkpoint_candidates(text: str) -> list[str]:
    out: list[str] = []
    for value in re.findall(r"['\"]([^'\"]+\.(?:ckpt|pth|pt|safetensors))['\"]", text):
        out.append(value)
    if "policy_best" in text and not out:
        out.append("{checkpoint_dir}/policy_best.ckpt")
    return out


def _first_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                return item
    return None


def _preferred_checkpoint_glob(
    *,
    user_value: str | None,
    generated_value: str | None,
    deep_candidates: Any,
) -> str | None:
    if user_value:
        return user_value
    deep_value = _best_checkpoint_candidate(deep_candidates)
    if generated_value and generated_value not in {"{checkpoint_dir}/*", "{checkpoint_dir}/*.*"}:
        return generated_value
    return deep_value or generated_value


def _best_checkpoint_candidate(value: Any) -> str | None:
    candidates: list[str] = []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [str(item) for item in value if isinstance(item, str) and item]
    for item in candidates:
        if "policy_best.ckpt" in item:
            return "{checkpoint_dir}/policy_best.ckpt"
    for item in candidates:
        if "checkpoint_dir" in item:
            return item
    for item in candidates:
        if any(item.endswith(suffix) for suffix in (".ckpt", ".pth", ".pt", ".safetensors")):
            return "{checkpoint_dir}/" + Path(item).name
    return None


def _collect_repo_snippets(
    repo: Path,
    files: list[str],
    commands: CommandIntake,
    *,
    max_files: int = 18,
    max_chars_per_file: int = 2400,
) -> list[dict[str, str]]:
    selected: list[str] = []

    def add(rel: str) -> None:
        if rel in selected:
            return
        path = repo / rel
        if path.is_file():
            selected.append(rel)

    for part in [
        *_command_parts(commands.dataset_command),
        *_command_parts(commands.train_command),
        *_command_parts(commands.eval_command),
    ]:
        if part.startswith("{") or part.startswith("-"):
            continue
        if (repo / part).is_file():
            add(part)

    for item in files:
        lower = item.lower()
        name = Path(item).name.lower()
        if name.startswith("readme"):
            add(item)
        elif name in {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "environment.yml"}:
            add(item)
        elif any(token in lower for token in ("train", "finetune", "dataset", "convert", "preprocess", "eval", "config")):
            add(item)
        if len(selected) >= max_files:
            break

    snippets: list[dict[str, str]] = []
    for rel in selected[:max_files]:
        path = repo / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        snippets.append(
            {
                "path": rel,
                "content": text[:max_chars_per_file],
                "truncated": str(len(text) > max_chars_per_file).lower(),
            }
        )
    return snippets


def _understanding_prompt(
    *,
    repo: Path,
    repo_summary: dict[str, Any],
    commands: CommandIntake,
    target_dataset_format: str,
    command_context: str,
    initial_profile: dict[str, Any],
    deep_inspection: dict[str, Any],
    snippets: list[dict[str, str]],
) -> str:
    payload = {
        "repo_root": str(repo),
        "repo_summary": repo_summary,
        "command_intake": {
            "dataset_command": _command_parts(commands.dataset_command),
            "train_command": _command_parts(commands.train_command),
            "eval_command": _command_parts(commands.eval_command),
            "env_keys": sorted(commands.env.keys()),
        },
        "user_target_dataset_format": target_dataset_format,
        "user_command_context": command_context,
        "initial_profile": initial_profile,
        "deep_repo_inspection": deep_inspection,
        "repo_snippets": snippets,
    }
    return (
        "RoboLineage needs to connect to this existing robot policy training framework without modifying "
        "the framework's own source code. The user gives two authoritative text boxes: the target "
        "training dataset format, and command/runtime context. Deeply infer the repository's data "
        "loader expectations, how RoboLineage raw rollouts should be adapted into that target dataset, how "
        "training should be launched, and where outputs probably live. Do not invent fields when "
        "the repository or user text does not support them.\n\n"
        "Return one JSON object with these fields:\n"
        "{\n"
        '  "confidence": 0.0-1.0,\n'
        '  "framework_type": "generic_policy|imitation_policy_like|vla_like|diffusion_policy_like|lerobot_like|...",\n'
        '  "repo_interpretation": "short explanation",\n'
        '  "target_dataset_interpretation": "normalized target dataset schema and required keys",\n'
        '  "dataset_input_expectation": "how the training repo consumes the generated dataset",\n'
        '  "adapter_plan": "how to convert RoboLineage rosbag2 raw rollouts into the target training dataset",\n'
        '  "training_entrypoint": "what train command appears to run",\n'
        '  "eval_entrypoint": "what eval command appears to run",\n'
        '  "checkpoint_expectation": "where checkpoints are written",\n'
        '  "tensorboard_logdir_expectation": "where TensorBoard event files are written, if any",\n'
        '  "metrics_source": "stdout|tensorboard|wandb|jsonl|unknown",\n'
        '  "eval_result_expectation": "where eval metrics/results are written",\n'
        '  "profile_patch": {\n'
        '    "outputs": {"checkpoint_glob": "...", "train_log": "...", "eval_result": "..."},\n'
        '    "log_patterns": {"step": "...", "loss": "...", "success_rate": "..."},\n'
        '    "staging": {"framework_input_dir": "..."}\n'
        "  },\n"
        '  "assumptions": ["..."],\n'
        '  "warnings": ["..."]\n'
        "}\n\n"
        "Use RoboLineage placeholders when useful: {framework_input_dir}, {selected_rollouts_dir}, "
        "{selected_rollouts_file}, {dataset_output}, {checkpoint_dir}, {eval_output}, "
        "{checkpoint_path}, {repo_root}. If unsure, leave a field absent instead of inventing.\n\n"
        f"Context JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _normalize_understanding(raw: dict[str, Any]) -> dict[str, Any]:
    patch = raw.get("profile_patch")
    if not isinstance(patch, dict):
        patch = {}
    out: dict[str, Any] = {
        "confidence": _clamped_confidence(raw.get("confidence")),
        "framework_type": _optional_stripped(raw.get("framework_type")),
        "repo_interpretation": _optional_stripped(raw.get("repo_interpretation")),
        "target_dataset_interpretation": _optional_stripped(raw.get("target_dataset_interpretation")),
        "dataset_input_expectation": _optional_stripped(raw.get("dataset_input_expectation")),
        "adapter_plan": _optional_stripped(raw.get("adapter_plan")),
        "training_entrypoint": _optional_stripped(raw.get("training_entrypoint")),
        "eval_entrypoint": _optional_stripped(raw.get("eval_entrypoint")),
        "checkpoint_expectation": _optional_stripped(raw.get("checkpoint_expectation")),
        "tensorboard_logdir_expectation": _optional_stripped(raw.get("tensorboard_logdir_expectation")),
        "metrics_source": _optional_stripped(raw.get("metrics_source")),
        "eval_result_expectation": _optional_stripped(raw.get("eval_result_expectation")),
        "profile_patch": {
            "outputs": _string_mapping(patch.get("outputs")),
            "log_patterns": _string_mapping(patch.get("log_patterns")),
            "staging": _string_mapping(patch.get("staging")),
        },
        "assumptions": _string_list(raw.get("assumptions")),
        "warnings": _string_list(raw.get("warnings")),
    }
    return {key: value for key, value in out.items() if value not in (None, {}, [])}


def _apply_understanding_to_profile(
    profile: dict[str, Any],
    understanding: dict[str, Any],
    *,
    user_supplied_framework_type: bool,
    user_supplied_outputs: dict[str, bool],
    user_supplied_fixed_input_dir: bool,
) -> None:
    if understanding.get("status") != "completed":
        return

    framework_type = _optional_stripped(understanding.get("framework_type"))
    if framework_type and not user_supplied_framework_type:
        profile["framework_type"] = framework_type

    patch = understanding.get("profile_patch") if isinstance(understanding.get("profile_patch"), dict) else {}
    outputs = patch.get("outputs") if isinstance(patch.get("outputs"), dict) else {}
    for key in ("checkpoint_glob", "train_log", "eval_result"):
        value = _optional_stripped(outputs.get(key))
        if value and not user_supplied_outputs.get(key, False):
            profile.setdefault("outputs", {})[key] = value

    staging = patch.get("staging") if isinstance(patch.get("staging"), dict) else {}
    framework_input_dir = _optional_stripped(staging.get("framework_input_dir"))
    if framework_input_dir and not user_supplied_fixed_input_dir:
        profile.setdefault("staging", {})["framework_input_dir"] = framework_input_dir

    log_patterns = patch.get("log_patterns") if isinstance(patch.get("log_patterns"), dict) else {}
    if log_patterns:
        profile.setdefault("log_patterns", {}).update(
            {str(key): str(value) for key, value in log_patterns.items() if str(value)}
        )


def _understanding_summary(
    understanding: dict[str, Any],
    *,
    understanding_path: Path | None,
    understanding_report_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": understanding.get("schema_version"),
        "status": understanding.get("status"),
        "used_llm": understanding.get("used_llm"),
        "model": understanding.get("model"),
        "confidence": understanding.get("confidence"),
        "framework_type": understanding.get("framework_type"),
        "repo_interpretation": understanding.get("repo_interpretation"),
        "target_dataset_interpretation": understanding.get("target_dataset_interpretation"),
        "adapter_plan": understanding.get("adapter_plan"),
        "training_entrypoint": understanding.get("training_entrypoint"),
        "checkpoint_expectation": understanding.get("checkpoint_expectation"),
        "tensorboard_logdir_expectation": understanding.get("tensorboard_logdir_expectation"),
        "metrics_source": understanding.get("metrics_source"),
        "assumptions": understanding.get("assumptions") or [],
        "warnings": understanding.get("warnings") or [],
        "understanding_path": str(understanding_path) if understanding_path is not None else None,
        "understanding_report_path": str(understanding_report_path)
        if understanding_report_path is not None
        else None,
    }


def _training_integration_manifest(
    *,
    profile: dict[str, Any],
    discovery: dict[str, Any],
    deep_inspection: dict[str, Any],
    understanding: dict[str, Any] | None,
) -> dict[str, Any]:
    outputs = dict(profile.get("outputs") or {})
    deep_outputs = deep_inspection.get("outputs") if isinstance(deep_inspection.get("outputs"), dict) else {}
    metrics_sources = deep_outputs.get("metrics_sources") if isinstance(deep_outputs.get("metrics_sources"), list) else []
    return {
        "schema_version": INTEGRATION_MANIFEST_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "name": discovery.get("name"),
        "framework_type": discovery.get("framework_type"),
        "repo": {
            "root": discovery.get("repo_root"),
            "summary": discovery.get("repo_summary") or {},
        },
        "dataset": {
            "target_contract": discovery.get("target_contract") or {},
            "adapter": discovery.get("dataset_adapter") or {},
            "adapter_registry": discovery.get("adapter_registry") or {},
        },
        "training": {
            "command": (discovery.get("command_intake") or {}).get("train_command") or [],
            "execution": discovery.get("execution") or {},
            "staging": discovery.get("staging") or {},
        },
        "outputs": {
            "stdout_capture": deep_outputs.get("stdout_capture") or "{workspace_dir}/train_command.log",
            "train_log": outputs.get("train_log"),
            "checkpoint_glob": outputs.get("checkpoint_glob"),
            "eval_result": outputs.get("eval_result"),
            "tensorboard_logdir_candidates": deep_outputs.get("tensorboard_logdir_candidates") or [],
            "checkpoint_glob_candidates": deep_outputs.get("checkpoint_glob_candidates") or [],
            "metrics_sources": metrics_sources,
            "primary_metrics_source": (
                understanding.get("metrics_source")
                if isinstance(understanding, dict) and understanding.get("metrics_source")
                else (metrics_sources[0] if metrics_sources else "unknown")
            ),
        },
        "deep_inspection": {
            "mode": deep_inspection.get("mode"),
            "command_entrypoint_files": deep_inspection.get("command_entrypoint_files") or [],
            "expanded_entrypoint_files": deep_inspection.get("expanded_entrypoint_files") or [],
            "files_read": deep_inspection.get("files_read") or [],
            "findings": deep_inspection.get("findings") or [],
        },
        "llm_understanding": _understanding_summary(
            understanding,
            understanding_path=None,
            understanding_report_path=None,
        )
        if isinstance(understanding, dict)
        else None,
        "warnings": discovery.get("warnings") or [],
    }


def _render_report(discovery: dict[str, Any]) -> str:
    command = discovery["command_intake"]
    summary = discovery["repo_summary"]
    staging = discovery["staging"]
    framework_input_dir = (
        staging.get("framework_input_dir")
        or staging.get("selected_rollouts_dir")
        or "(dataset_output)"
    )
    lines = [
        "# Framework Discovery Report",
        "",
        f"- name: `{discovery['name']}`",
        f"- framework_type: `{discovery['framework_type']}`",
        f"- repo_root: `{discovery['repo_root']}`",
        f"- created_at: `{discovery['created_at']}`",
        "",
        "## User Intake",
        "",
        f"- target_dataset_format_chars: `{len((discovery.get('agent_intake') or {}).get('target_dataset_format') or '')}`",
        f"- command_context_chars: `{len((discovery.get('agent_intake') or {}).get('command_context') or '')}`",
        "",
        "## Commands",
        "",
        f"- dataset: `{_join(command['dataset_command']) or '(missing)'}`",
        f"- train: `{_join(command['train_command']) or '(missing)'}`",
        f"- eval: `{_join(command['eval_command']) or '(missing)'}`",
        "",
        "## RoboLineage Staging",
        "",
        f"- selected_rollouts_file: `{staging['selected_rollouts_file']}`",
        f"- selected_rollouts_dir: `{staging.get('selected_rollouts_dir') or '(not created unless needed)'}`",
        f"- framework_input_dir: `{framework_input_dir}`",
        "",
        "## Target Data Contract",
        "",
    ]
    target_contract = discovery.get("target_contract") if isinstance(discovery.get("target_contract"), dict) else {}
    adapter_registry = discovery.get("adapter_registry") if isinstance(discovery.get("adapter_registry"), dict) else {}
    selected_adapter = adapter_registry.get("selected_adapter") if isinstance(adapter_registry.get("selected_adapter"), dict) else {}
    lines.extend(
        [
            f"- dataset_kind: `{target_contract.get('dataset_kind') or '(unknown)'}`",
            f"- input_path: `{target_contract.get('input_path_template') or '{dataset_output}'}`",
            f"- camera_names: `{', '.join(target_contract.get('camera_names') or []) or '(none inferred)'}`",
            f"- adapter: `{selected_adapter.get('adapter_id') or (discovery.get('dataset_adapter') or {}).get('adapter_id') or '(none)'}`",
            f"- adapter_strategy: `{selected_adapter.get('strategy') or (discovery.get('dataset_adapter') or {}).get('strategy') or '(none)'}`",
            "",
            "## Training Integration Outputs",
            "",
        ]
    )
    manifest = discovery.get("integration_manifest") if isinstance(discovery.get("integration_manifest"), dict) else {}
    manifest_outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    lines.extend(
        [
            f"- primary_metrics_source: `{manifest_outputs.get('primary_metrics_source') or '(unknown)'}`",
            f"- tensorboard_logdir_candidates: `{', '.join(manifest_outputs.get('tensorboard_logdir_candidates') or []) or '(none)'}`",
            f"- checkpoint_glob: `{manifest_outputs.get('checkpoint_glob') or '(unknown)'}`",
            f"- stdout_capture: `{manifest_outputs.get('stdout_capture') or '{workspace_dir}/train_command.log'}`",
            "",
            "## Discovery Events",
            "",
        ]
    )
    for event in discovery.get("events") or []:
        if isinstance(event, dict):
            lines.append(
                f"- `{event.get('event')}`"
                f" `{event.get('created_at')}`"
            )
    if not discovery.get("events"):
        lines.append("- `(none)`")
    lines.extend([
        "",
        "## Candidate Files",
        "",
        f"- dataset scripts: `{', '.join(summary['candidate_dataset_scripts']) or '(none)'}`",
        f"- train scripts: `{', '.join(summary['candidate_train_scripts']) or '(none)'}`",
        f"- eval scripts: `{', '.join(summary['candidate_eval_scripts']) or '(none)'}`",
        "",
    ])
    understanding = discovery.get("llm_understanding") if isinstance(discovery.get("llm_understanding"), dict) else None
    if understanding is not None:
        lines.extend(
            [
                "## LLM Understanding",
                "",
                f"- status: `{understanding.get('status')}`",
                f"- used_llm: `{understanding.get('used_llm')}`",
                f"- confidence: `{understanding.get('confidence')}`",
                f"- framework_type: `{understanding.get('framework_type')}`",
                f"- report: `{understanding.get('understanding_report_path') or '(none)'}`",
                "",
            ]
        )
    lines.extend(["## Warnings", ""])
    warnings = discovery.get("warnings") or []
    lines.extend([f"- `{item}`" for item in warnings] or ["- `(none)`"])
    lines.append("")
    return "\n".join(lines)


def _render_understanding_report(understanding: dict[str, Any]) -> str:
    patch = understanding.get("profile_patch") if isinstance(understanding.get("profile_patch"), dict) else {}
    outputs = patch.get("outputs") if isinstance(patch.get("outputs"), dict) else {}
    log_patterns = patch.get("log_patterns") if isinstance(patch.get("log_patterns"), dict) else {}
    staging = patch.get("staging") if isinstance(patch.get("staging"), dict) else {}
    lines = [
        "# Framework Understanding",
        "",
        f"- status: `{understanding.get('status')}`",
        f"- used_llm: `{understanding.get('used_llm')}`",
        f"- model: `{understanding.get('model') or '(none)'}`",
        f"- confidence: `{understanding.get('confidence')}`",
        f"- framework_type: `{understanding.get('framework_type') or '(unknown)'}`",
        f"- repo_interpretation: {understanding.get('repo_interpretation') or '(none)'}",
        "",
        "## Expected Flow",
        "",
        f"- target_dataset: {understanding.get('target_dataset_interpretation') or '(unknown)'}",
        f"- dataset_input: {understanding.get('dataset_input_expectation') or '(unknown)'}",
        f"- adapter_plan: {understanding.get('adapter_plan') or '(unknown)'}",
        f"- training_entrypoint: {understanding.get('training_entrypoint') or '(unknown)'}",
        f"- eval_entrypoint: {understanding.get('eval_entrypoint') or '(unknown)'}",
        f"- checkpoint: {understanding.get('checkpoint_expectation') or '(unknown)'}",
        f"- eval_result: {understanding.get('eval_result_expectation') or '(unknown)'}",
        "",
        "## Profile Patch",
        "",
        f"- outputs: `{json.dumps(outputs, ensure_ascii=False)}`",
        f"- log_patterns: `{json.dumps(log_patterns, ensure_ascii=False)}`",
        f"- staging: `{json.dumps(staging, ensure_ascii=False)}`",
        "",
        "## Assumptions",
        "",
    ]
    lines.extend([f"- {item}" for item in understanding.get("assumptions") or []] or ["- (none)"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- `{item}`" for item in understanding.get("warnings") or []] or ["- `(none)`"])
    lines.append("")
    return "\n".join(lines)


def _join(parts: list[str] | tuple[str, ...]) -> str:
    return " ".join(str(part) for part in parts)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _clamped_confidence(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, num))


def _optional_stripped(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(item).strip()
        for key, item in value.items()
        if item is not None and str(item).strip()
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "training_framework"


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
