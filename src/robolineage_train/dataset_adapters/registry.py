from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA_VERSION = "RoboLineage.dataset_contract.v1"
ADAPTER_REGISTRY_SCHEMA_VERSION = "RoboLineage.dataset_adapter_registry.v1"


@dataclass(frozen=True)
class DataField:
    path: str
    role: str
    required: bool = True
    shape: str | None = None
    dtype: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetDataContract:
    """What the target training code actually consumes."""

    framework_type: str
    dataset_kind: str
    input_path_template: str = "{dataset_output}"
    fields: tuple[DataField, ...] = ()
    camera_names: tuple[str, ...] = ()
    required_raw_capabilities: tuple[str, ...] = ()
    optional_raw_capabilities: tuple[str, ...] = ()
    evidence_files: tuple[str, ...] = ()
    confidence: float = 0.5
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fields"] = [field.to_dict() for field in self.fields]
        return data


@dataclass(frozen=True)
class AdapterCandidate:
    adapter_id: str
    strategy: str
    target_dataset_kind: str
    source_data_policy: str
    confidence: float
    supported_raw_capabilities: tuple[str, ...]
    generated: bool = False
    module: str | None = None
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RAW_ROSBAG_CAPABILITIES = (
    "raw.rosbag2.camera_compressed_image",
    "raw.rosbag2.robot_state_vector",
)


def infer_target_data_contract(
    *,
    repo_root: Path,
    repo_files: list[str],
    target_dataset_format: str,
    command_context: str,
    framework_type: str,
) -> TargetDataContract:
    """Infer the minimum dataset contract the training pipeline appears to use.

    This deliberately models the consumed data, not a full desired schema. RoboLineage
    raw data can be a strict subset of an external repo's broader format as
    long as the fields used by the configured training command are materialized.
    """

    repo = Path(repo_root)
    text = _lower_blob(target_dataset_format, command_context, _read_repo_text(repo, repo_files))
    evidence = _evidence_files(repo_files)
    camera_names = tuple(_target_camera_names(target_dataset_format, command_context, text))

    if _looks_like_act_hdf5(text):
        fields = (
            DataField("/action", "action", shape="(T, 14)", source="raw.rosbag2.robot_state_vector"),
            DataField("/observations/qpos", "state", shape="(T, 14)", source="raw.rosbag2.robot_state_vector"),
            DataField("/observations/qvel", "state", shape="(T, 14)", source="raw.rosbag2.robot_state_vector"),
            DataField("/observations/eef", "state", shape="(T, 14)", source="raw.rosbag2.robot_state_vector"),
            DataField("/observations/effort", "state", shape="(T, 14)", source="raw.rosbag2.robot_state_vector"),
            DataField("/observations/images/{camera}", "image", source="raw.rosbag2.camera_compressed_image"),
            DataField("/action_eef", "action", shape="(T, 14)", source="raw.rosbag2.robot_state_vector"),
            DataField("/action_base", "action", shape="(T, 6)", required=False, source="default.zeros"),
            DataField("/action_velocity", "action", shape="(T, 4)", required=False, source="default.zeros"),
            DataField("/observations/robot_base", "state", shape="(T, 6)", required=False, source="default.zeros"),
            DataField("/observations/base_velocity", "state", shape="(T, 4)", required=False, source="default.zeros"),
        )
        return TargetDataContract(
            framework_type=framework_type,
            dataset_kind="act_hdf5",
            fields=fields,
            camera_names=camera_names,
            required_raw_capabilities=RAW_ROSBAG_CAPABILITIES,
            optional_raw_capabilities=("raw.robot_base", "raw.base_velocity", "raw.action_commands"),
            evidence_files=evidence,
            confidence=0.86,
            assumptions=(
                "Training loader consumes contiguous episode_*.hdf5 files from DATASET_DIR or --datasets.",
                "Only fields consumed by the configured training flags need to be faithful.",
                "Robot state/action values are driver-native passthrough: column order is stable and no unit or coordinate conversion is applied.",
            ),
            warnings=(
                "Base/head velocity fields are zero-filled unless RoboLineage raw recorder captures those streams.",
            ),
        )

    if "{selected_rollouts_file}" in command_context:
        return TargetDataContract(
            framework_type=framework_type,
            dataset_kind="selected_rollouts_manifest",
            input_path_template="{selected_rollouts_file}",
            fields=(DataField("selected_rollouts.json", "manifest", source="RoboLineage.selection"),),
            evidence_files=evidence,
            confidence=0.7,
            assumptions=("Training command directly consumes RoboLineage selected_rollouts.json.",),
        )

    return TargetDataContract(
        framework_type=framework_type,
        dataset_kind="unknown_custom",
        fields=(),
        camera_names=camera_names,
        evidence_files=evidence,
        confidence=0.25,
        warnings=(
            "Could not infer a concrete dataset contract from repo code and command context.",
            "Provide dataset details or a dataset command for this training pipeline.",
        ),
    )


def choose_adapter_candidate(contract: TargetDataContract) -> AdapterCandidate:
    if contract.dataset_kind == "act_hdf5":
        return AdapterCandidate(
            adapter_id="rosbag_act_hdf5",
            strategy="registered_adapter_module",
            target_dataset_kind=contract.dataset_kind,
            source_data_policy="read_only",
            confidence=min(0.9, contract.confidence),
            supported_raw_capabilities=RAW_ROSBAG_CAPABILITIES,
            module="robolineage_train.dataset_adapters.rosbag_act_hdf5",
            assumptions=(
                "Materialize ACT HDF5 directly from rosbag2 raw topics on a fixed 30 Hz training timeline.",
                "Preserve driver-native robot state/action values; resample over time only.",
            ),
            warnings=contract.warnings,
        )
    if contract.dataset_kind == "selected_rollouts_manifest":
        return AdapterCandidate(
            adapter_id="direct_selected_rollouts_file",
            strategy="direct_manifest",
            target_dataset_kind=contract.dataset_kind,
            source_data_policy="read_only",
            confidence=contract.confidence,
            supported_raw_capabilities=(),
            assumptions=("No dataset materialization is required.",),
        )
    return AdapterCandidate(
        adapter_id="unresolved_custom_dataset",
        strategy="requires_dataset_adapter",
        target_dataset_kind=contract.dataset_kind,
        source_data_policy="read_only",
        confidence=contract.confidence,
        supported_raw_capabilities=(),
        warnings=contract.warnings,
    )


def registry_summary_payload(
    *,
    contract: TargetDataContract,
    candidate: AdapterCandidate,
) -> dict[str, Any]:
    return {
        "schema_version": ADAPTER_REGISTRY_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_contract": contract.to_dict(),
        "selected_adapter": candidate.to_dict(),
    }


def _looks_like_act_hdf5(text: str) -> bool:
    has_hdf5 = "hdf5" in text or ".hdf5" in text
    has_episode = "episode_" in text or re.search(r"\bepisode[-_a-z0-9*]*\.hdf5\b", text) is not None
    has_observation_state = (
        "observations/qpos" in text
        or "/observations/qpos" in text
        or ("observations/" in text and re.search(r"(?<![a-z0-9_])qpos(?![a-z0-9_])", text) is not None)
    )
    has_action = "action_eef" in text or "/action" in text or re.search(r"\baction\b", text) is not None
    return has_hdf5 and has_episode and has_observation_state and has_action


def _target_camera_names(target_dataset_format: str, command_context: str, text: str) -> list[str]:
    explicit = _camera_names_from_command_context(command_context, semantic="hdf5" in f"{target_dataset_format}\n{text}".lower())
    if explicit:
        return explicit
    source = f"{target_dataset_format}\n{command_context}\n{text}"
    known = ("head", "left_wrist", "right_wrist", "camera_h", "camera_l", "camera_r")
    found = [name for name in known if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", source)]
    if not found:
        return ["head", "right_wrist"] if "hdf5" in source else []
    aliases = {"camera_h": "head", "camera_l": "left_wrist", "camera_r": "right_wrist"}
    normalized: list[str] = []
    for name in found:
        value = aliases.get(name, name) if "hdf5" in source else name
        if value not in normalized:
            normalized.append(value)
    return normalized


def _camera_names_from_command_context(command_context: str, *, semantic: bool) -> list[str]:
    aliases = {"camera_h": "head", "camera_l": "left_wrist", "camera_r": "right_wrist"} if semantic else {}
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
        out: list[str] = []
        for token in (item.strip().lower() for item in re.split(r"[,\s]+", match.group(1))):
            if token not in known:
                continue
            value = aliases.get(token, token)
            if value not in out:
                out.append(value)
        if out:
            return out
    return []


def _read_repo_text(repo: Path, repo_files: list[str], max_chars: int = 160000) -> str:
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
        chunk = text[:remaining]
        parts.append(f"\n# file: {rel}\n{chunk}")
        remaining -= len(chunk)
    return "\n".join(parts)


def _evidence_files(repo_files: list[str]) -> tuple[str, ...]:
    selected = []
    for rel in repo_files:
        lower = rel.lower()
        if any(token in lower for token in ("02_train", "train.py", "utils.py", "collect.py", "dataset")):
            selected.append(rel)
        if len(selected) >= 12:
            break
    return tuple(selected)


def _lower_blob(*parts: str) -> str:
    return "\n".join(str(part or "") for part in parts).lower()
