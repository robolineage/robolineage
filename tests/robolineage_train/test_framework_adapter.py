from __future__ import annotations

import json
import sys
import csv
import base64
from pathlib import Path

import pytest

from robolineage_train import (
    CommandIntake,
    FrameworkAdapter,
    FrameworkDiscoveryAgent,
    load_framework_profile,
    OpenAICompatibleDiscoveryClient,
    parse_training_log,
    write_selected_rollouts,
)
from robolineage_train.__main__ import main
from robolineage_train.discovery import _preferred_checkpoint_glob, _resolve_command_files, _scan_repo
from robolineage_schemas import validate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _assert_schema_valid(path: Path, schema_name: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = [issue for issue in validate(payload, schema_name) if issue.severity == "error"]
    assert errors == []


def _rollout(
    root: Path,
    rollout_id: str,
    *,
    decision: str,
    final_success: bool = True,
    accepted_for_training: bool | None = None,
) -> Path:
    path = root / rollout_id
    path.mkdir(parents=True)
    admission = {
        "decision": decision,
        "data_use": ["success_trajectory"],
        "task_description": "pick object",
    }
    if accepted_for_training is not None:
        admission["accepted_for_training"] = accepted_for_training
    _write_json(path / "dataset_admission.json", admission)
    _write_json(
        path / "rollout_summary.json",
        {
            "rollout_id": rollout_id,
            "final_success": final_success,
            "task_description": "pick object",
        },
    )
    _write_json(path / "failure_analysis.json", {"candidate_count": 0})
    return path


def _write_raw_rosbag_rollout(raw: Path, frame_count: int = 4) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    raw.mkdir(parents=True)
    bag_dir = raw / "rosbag2"
    bag_dir.mkdir(parents=True)
    ramp = np.linspace(0.0, 1.0, frame_count, dtype=np.float32).reshape(-1, 1)
    left_state = np.zeros((frame_count, 27), dtype=np.float32)
    left_state[:, :7] = ramp
    left_state[:, 7:14] = 0.2
    left_state[:, 14:21] = 0.3
    left_state[:, 21:27] = 0.4
    right_state = np.zeros((frame_count, 27), dtype=np.float32)
    right_state[:, :7] = 10.0 + 2.0 * ramp
    right_state[:, 7:14] = 0.7
    right_state[:, 14:21] = 0.8
    right_state[:, 21:27] = 0.9

    def jpeg_b64(value: int) -> str:
        ok, encoded = cv2.imencode(".jpg", np.full((6, 8, 3), value, dtype=np.uint8))
        assert ok
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    with (bag_dir / "messages.jsonl").open("w", encoding="utf-8") as f:
        for idx in range(frame_count):
            stamp_ns = idx * 100_000_000
            rows = [
                {
                    "topic": "/cam/head/image/compressed",
                    "stamp_ns": stamp_ns,
                    "type": "sensor_msgs/msg/CompressedImage",
                    "data": {"format": "jpeg", "bytes_b64": jpeg_b64(10 + idx)},
                },
                {
                    "topic": "/cam/right_wrist/image/compressed",
                    "stamp_ns": stamp_ns,
                    "type": "sensor_msgs/msg/CompressedImage",
                    "data": {"format": "jpeg", "bytes_b64": jpeg_b64(50 + idx)},
                },
                {
                    "topic": "/arm/left/state",
                    "stamp_ns": stamp_ns,
                    "type": "RoboLineage/test/RobotStateVector",
                    "data": {"vector": left_state[idx].tolist()},
                },
                {
                    "topic": "/arm/right/state",
                    "stamp_ns": stamp_ns,
                    "type": "RoboLineage/test/RobotStateVector",
                    "data": {"vector": right_state[idx].tolist()},
                },
            ]
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    _write_json(
        raw / "raw_manifest.json",
        {
            "schema_version": "RoboLineage.raw_rosbag_manifest.v1",
            "status": "closed",
            "bag_dir": str(bag_dir),
            "topics": [
                "/cam/head/image/compressed",
                "/cam/right_wrist/image/compressed",
                "/arm/left/state",
                "/arm/right/state",
            ],
        },
    )


def _write_host_repo(repo: Path) -> None:
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "build_dataset.py").write_text(
        "\n".join(
            [
                "import json, sys",
                "from pathlib import Path",
                "selected = json.loads(Path(sys.argv[1]).read_text())",
                "out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)",
                "(out / 'dataset.json').write_text(json.dumps({'count': selected['selected_rollout_count']}))",
                "print('dataset_count=' + str(selected['selected_rollout_count']))",
            ]
        ),
        encoding="utf-8",
    )
    (scripts / "train.py").write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "dataset = Path(sys.argv[1])",
                "out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)",
                "(out / 'policy.ckpt').write_text('checkpoint')",
                "(out / 'train.log').write_text('step=17 loss=0.125\\n')",
                "print('step=17 loss=0.125')",
                "print('saved checkpoint to ' + str(out / 'policy.ckpt'))",
            ]
        ),
        encoding="utf-8",
    )
    (scripts / "eval.py").write_text(
        "\n".join(
            [
                "import json, sys",
                "from pathlib import Path",
                "checkpoint = Path(sys.argv[1])",
                "out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)",
                "(out / 'result.json').write_text(json.dumps({'success_rate': 0.8, 'checkpoint': str(checkpoint)}))",
                "print('{\"success_rate\": 0.8}')",
            ]
        ),
        encoding="utf-8",
    )


def _profile(path: Path, repo: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "name: generic_policy_stack",
                "framework_type: generic",
                "adapter_version: '0.1'",
                f"repo_root: {repo}",
                "dataset_command:",
                f"  args: [{json.dumps(sys.executable)}, scripts/build_dataset.py, '{{selected_rollouts_file}}', '{{dataset_output}}']",
                "train_command:",
                f"  args: [{json.dumps(sys.executable)}, scripts/train.py, '{{dataset_output}}', '{{checkpoint_dir}}']",
                "eval_command:",
                f"  args: [{json.dumps(sys.executable)}, scripts/eval.py, '{{checkpoint_path}}', '{{eval_output}}']",
                "outputs:",
                "  checkpoint_glob: '{checkpoint_dir}/*.ckpt'",
                "  train_log: '{checkpoint_dir}/train.log'",
                "  eval_result: '{eval_output}/result.json'",
                "log_patterns:",
                "  step: 'step=(\\d+)'",
                "  loss: 'loss=([0-9.]+)'",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_write_selected_rollouts_creates_json_and_symlink_dir(tmp_path):
    rollouts = tmp_path / "rollouts"
    accepted = _rollout(rollouts, "accepted_a", decision="accepted")
    _rollout(rollouts, "review_b", decision="needs_review")
    output = tmp_path / "selected.json"
    selected_dir = tmp_path / "selected"

    rows = write_selected_rollouts(
        rollouts_root=rollouts,
        output_path=output,
        dataset_version="v3",
        selected_dir=selected_dir,
    )

    assert [row.rollout_id for row in rows] == ["accepted_a"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == "v3"
    assert payload["selected_rollout_count"] == 1
    assert (selected_dir / "accepted_a").resolve() == accepted.resolve()


def test_write_selected_rollouts_can_filter_frontend_selection(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "accepted_a", decision="accepted")
    selected = _rollout(rollouts, "accepted_b", decision="accepted")
    _rollout(rollouts, "review_c", decision="needs_review")
    output = tmp_path / "selected.json"
    selected_dir = tmp_path / "selected"

    rows = write_selected_rollouts(
        rollouts_root=rollouts,
        output_path=output,
        dataset_version="v4",
        selected_dir=selected_dir,
        include_decisions=("accepted", "needs_review"),
        include_rollout_ids=("accepted_b",),
    )

    assert [row.rollout_id for row in rows] == ["accepted_b"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_rollout_count"] == 1
    assert payload["selected_rollouts"][0]["rollout_id"] == "accepted_b"
    assert (selected_dir / "accepted_b").resolve() == selected.resolve()
    assert not (selected_dir / "accepted_a").exists()


def test_write_selected_rollouts_uses_training_eligibility_before_review_bucket(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "clean", decision="accepted", accepted_for_training=True)
    _rollout(rollouts, "uncertain_success", decision="needs_review", accepted_for_training=True)
    _rollout(rollouts, "failure_pool", decision="needs_review", accepted_for_training=False)
    output = tmp_path / "selected.json"

    rows = write_selected_rollouts(
        rollouts_root=rollouts,
        output_path=output,
        dataset_version="v5",
        include_decisions=("accepted", "needs_review"),
    )

    assert [row.rollout_id for row in rows] == ["clean", "uncertain_success"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [row["rollout_id"] for row in payload["selected_rollouts"]] == [
        "clean",
        "uncertain_success",
    ]


def test_framework_adapter_runs_existing_dataset_train_eval_commands(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "a", decision="accepted")
    _rollout(rollouts, "b", decision="needs_review")
    repo = tmp_path / "host_repo"
    _write_host_repo(repo)
    profile = load_framework_profile(_profile(tmp_path / "framework.yaml", repo))

    result = FrameworkAdapter(profile).run(
        rollouts_root=rollouts,
        workspace_dir=tmp_path / "ROBOLINEAGE_run",
        dataset_version="v1",
        policy_version="1.0.0",
    )

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.name == "policy.ckpt"
    assert result.eval_result_path is not None
    assert result.eval_result_path.exists()
    status = json.loads(result.training_status_path.read_text(encoding="utf-8"))
    assert status["metrics"]["step"] == 17
    assert status["metrics"]["loss"] == 0.125
    training_result = json.loads(result.training_result_path.read_text(encoding="utf-8"))
    assert training_result["framework"] == "generic_policy_stack"
    assert training_result["selected_rollout_count"] == 1
    assert training_result["selected_rollouts_dir"] is None
    assert result.selected_rollouts_dir is None
    _assert_schema_valid(result.training_status_path, "training_status")
    _assert_schema_valid(result.training_result_path, "training_result")


def test_framework_adapter_can_adapt_dataset_before_training(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "a", decision="accepted")
    repo = tmp_path / "host_repo"
    _write_host_repo(repo)
    profile = load_framework_profile(_profile(tmp_path / "framework.yaml", repo))
    workspace = tmp_path / "ROBOLINEAGE_run"

    adapted = FrameworkAdapter(profile).adapt_dataset(
        rollouts_root=rollouts,
        workspace_dir=workspace,
        dataset_version="v1",
        policy_version="1.0.0",
    )

    assert (adapted.dataset_output_dir / "dataset.json").exists()
    assert adapted.dataset_adapt_status_path.exists()
    assert not (workspace / "training_result.json").exists()
    adapt_status = json.loads(adapted.dataset_adapt_status_path.read_text(encoding="utf-8"))
    assert adapt_status["status"] == "completed"

    result = FrameworkAdapter(profile).run_training_only(
        workspace_dir=workspace,
        dataset_version="v1",
        policy_version="1.0.0",
    )

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.name == "policy.ckpt"
    assert result.training_result_path.exists()


def test_framework_adapter_creates_symlinks_when_command_uses_selected_rollouts_dir(tmp_path):
    rollouts = tmp_path / "rollouts"
    accepted = _rollout(rollouts, "a", decision="accepted")
    repo = tmp_path / "host_repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "build_dataset.py").write_text(
        "\n".join(
            [
                "import json, sys",
                "from pathlib import Path",
                "selected_dir = Path(sys.argv[1])",
                "out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)",
                "(out / 'dataset.json').write_text(json.dumps({'rollouts': sorted(p.name for p in selected_dir.iterdir())}))",
            ]
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "framework.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "name: selected_dir_framework",
                f"repo_root: {repo}",
                "dataset_command:",
                f"  args: [{json.dumps(sys.executable)}, scripts/build_dataset.py, '{{selected_rollouts_dir}}', '{{dataset_output}}']",
            ]
        ),
        encoding="utf-8",
    )

    result = FrameworkAdapter(load_framework_profile(profile_path)).run(
        rollouts_root=rollouts,
        workspace_dir=tmp_path / "ROBOLINEAGE_run",
        dataset_version="v1",
        policy_version="1.0.0",
    )

    assert result.selected_rollouts_dir is not None
    assert (result.selected_rollouts_dir / "a").resolve() == accepted.resolve()
    dataset = json.loads((result.dataset_output_dir / "dataset.json").read_text(encoding="utf-8"))
    assert dataset["rollouts"] == ["a"]


def test_framework_run_cli(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "a", decision="accepted")
    repo = tmp_path / "host_repo"
    _write_host_repo(repo)
    profile = _profile(tmp_path / "framework.yaml", repo)
    workspace = tmp_path / "ROBOLINEAGE_run"

    assert main([
        "framework-run",
        "--profile", str(profile),
        "--rollouts-root", str(rollouts),
        "--workspace", str(workspace),
        "--dataset-version", "v1",
        "--policy-version", "1.0.0",
    ]) == 0

    assert (workspace / "staging" / "selected_rollouts.json").exists()
    assert (workspace / "training_result.json").exists()


def test_discovery_agent_generates_profile_and_adapter_uses_framework_input_symlinks(tmp_path):
    rollouts = tmp_path / "rollouts"
    accepted = _rollout(rollouts, "a", decision="accepted")
    repo = tmp_path / "framework_input_repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (repo / "README.md").write_text("# Policy training repo\n", encoding="utf-8")
    (scripts / "build_dataset.py").write_text(
        "\n".join(
            [
                "import json, sys",
                "from pathlib import Path",
                "input_dir = Path(sys.argv[1])",
                "out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)",
                "rollouts = sorted(p.name for p in input_dir.iterdir())",
                "(out / 'dataset.json').write_text(json.dumps({'rollouts': rollouts}))",
                "print('dataset_count=' + str(len(rollouts)))",
            ]
        ),
        encoding="utf-8",
    )
    (scripts / "train.py").write_text(
        "\n".join(
            [
                "import json, sys",
                "from pathlib import Path",
                "out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)",
                "ckpt = out / 'policy.ckpt'; ckpt.write_text('policy')",
                "print(json.dumps({'step': 5, 'loss': 0.3, 'checkpoint': str(ckpt)}))",
            ]
        ),
        encoding="utf-8",
    )

    discovery = FrameworkDiscoveryAgent().discover(
        repo_root=repo,
        output_dir=tmp_path / "profiles" / "framework_input",
        target_dataset_format=(
            "Format: directory package\n"
            "Input manifest: selected_rollouts.json\n"
            "Output: dataset.json"
        ),
        command_context=(
            f"repo_root: {repo}\n"
            f"train command: {sys.executable} scripts/train.py {{dataset_output}} {{checkpoint_dir}}"
        ),
        commands=CommandIntake(
            dataset_command=f"{sys.executable} scripts/build_dataset.py {{framework_input_dir}} {{dataset_output}}",
            train_command=f"{sys.executable} scripts/train.py {{dataset_output}} {{checkpoint_dir}}",
        ),
    )
    profile = load_framework_profile(discovery.profile_path)

    result = FrameworkAdapter(profile).run(
        rollouts_root=rollouts,
        workspace_dir=tmp_path / "ROBOLINEAGE_run",
        dataset_version="v1",
        policy_version="1.0.0",
    )

    assert profile.framework_type == "generic_policy"
    assert (tmp_path / "profiles" / "framework_input" / "framework_discovery.json").exists()
    assert result.framework_input_dir is not None
    assert (result.framework_input_dir / "a").resolve() == accepted.resolve()
    dataset = json.loads((result.dataset_output_dir / "dataset.json").read_text(encoding="utf-8"))
    assert dataset["rollouts"] == ["a"]
    discovery_payload = json.loads(discovery.discovery_path.read_text(encoding="utf-8"))
    assert "selected_rollouts.json" in discovery_payload["agent_intake"]["target_dataset_format"]


class _FakeDiscoveryLLM:
    model = "fake-discovery-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "confidence": 0.87,
                "framework_type": "vla_like",
                "repo_interpretation": "VLA fine-tuning repo with JSONL training logs",
                "dataset_input_expectation": "dataset script consumes {framework_input_dir}",
                "training_entrypoint": "scripts/train_vla.py",
                "eval_entrypoint": "scripts/eval_policy.py",
                "checkpoint_expectation": "checkpoints are written under {checkpoint_dir}",
                "eval_result_expectation": "eval writes metrics.json in {eval_output}",
                "profile_patch": {
                    "outputs": {
                        "checkpoint_glob": "{checkpoint_dir}/ckpt_*.safetensors",
                        "train_log": "{checkpoint_dir}/metrics.jsonl",
                        "eval_result": "{eval_output}/metrics.json",
                    },
                    "log_patterns": {
                        "step": '"step":\\s*(\\d+)',
                        "loss": '"loss":\\s*([0-9.]+)',
                        "success_rate": '"success_rate":\\s*([0-9.]+)',
                    },
                    "staging": {"framework_input_dir": "{staging_dir}/vla_rollouts"},
                },
                "assumptions": ["commands remain authoritative"],
                "warnings": ["confirm checkpoint extension with framework owner"],
            }
        )


def test_framework_discovery_llm_understanding_refines_profile_without_rewriting_commands(tmp_path):
    repo = tmp_path / "unknown_policy_repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (repo / "README.md").write_text("# VLA policy training\n", encoding="utf-8")
    (scripts / "make_dataset.py").write_text("print('build dataset')\n", encoding="utf-8")
    (scripts / "train_vla.py").write_text("print('{\"step\": 1, \"loss\": 0.4}')\n", encoding="utf-8")
    (scripts / "eval_policy.py").write_text("print('{\"success_rate\": 0.7}')\n", encoding="utf-8")
    fake = _FakeDiscoveryLLM()

    discovery = FrameworkDiscoveryAgent(llm_client=fake).discover(
        repo_root=repo,
        output_dir=tmp_path / "profiles" / "unknown",
        commands=CommandIntake(
            dataset_command="python scripts/make_dataset.py --input {framework_input_dir}",
            train_command="python scripts/train_vla.py --out {checkpoint_dir}",
            eval_command="python scripts/eval_policy.py --ckpt {checkpoint_path}",
        ),
        enable_llm_understanding=True,
    )

    assert fake.prompts
    assert discovery.framework_type == "vla_like"
    assert discovery.understanding_path is not None
    assert discovery.understanding_path.exists()
    assert discovery.understanding_report_path is not None
    assert discovery.understanding_report_path.exists()
    profile = load_framework_profile(discovery.profile_path)
    assert profile.framework_type == "vla_like"
    assert profile.outputs.checkpoint_glob == "{checkpoint_dir}/ckpt_*.safetensors"
    assert profile.outputs.train_log == "{checkpoint_dir}/metrics.jsonl"
    assert profile.outputs.eval_result == "{eval_output}/metrics.json"
    assert profile.staging.framework_input_dir == "{staging_dir}/vla_rollouts"
    assert profile.log_patterns["success_rate"] == '"success_rate":\\s*([0-9.]+)'
    assert profile.dataset_command is not None
    assert profile.dataset_command.args == (
        "python",
        "scripts/make_dataset.py",
        "--input",
        "{framework_input_dir}",
    )
    discovery_payload = json.loads(discovery.discovery_path.read_text(encoding="utf-8"))
    assert discovery_payload["llm_understanding"]["status"] == "completed"
    assert any("confirm checkpoint extension" in item for item in discovery_payload["warnings"])


def test_discovery_prioritizes_repo_code_over_agent_memory_for_remote_style_commands(tmp_path):
    repo = tmp_path / "policy_repo"
    (repo / ".agent" / "context").mkdir(parents=True)
    for idx in range(300):
        (repo / ".agent" / "context" / f"note_{idx}.md").write_text("checkpoint dataset\n", encoding="utf-8")
    (repo / "tools").mkdir()
    (repo / "tools" / "train.sh").write_text("#!/usr/bin/env bash\npython act/train.py\n", encoding="utf-8")
    (repo / "act").mkdir()
    (repo / "act" / "train.py").write_text("print('train')\n", encoding="utf-8")

    files = _scan_repo(repo)

    assert not any(item.startswith(".agent/") for item in files)
    assert "tools/train.sh" in files
    assert _resolve_command_files(repo, CommandIntake(train_command="/tools/train.sh")) == ["tools/train.sh"]


def test_discovery_prefers_policy_best_over_broad_checkpoint_glob():
    assert _preferred_checkpoint_glob(
        user_value=None,
        generated_value="{checkpoint_dir}/*",
        deep_candidates=["policy_best.ckpt", "policy_latest_seed_{seed}.ckpt"],
    ) == "{checkpoint_dir}/policy_best.ckpt"
    assert _preferred_checkpoint_glob(
        user_value="{checkpoint_dir}/custom.ckpt",
        generated_value="{checkpoint_dir}/*",
        deep_candidates=["policy_best.ckpt"],
    ) == "{checkpoint_dir}/custom.ckpt"


def test_discovery_client_prefers_discovery_route_env_over_other_llm_routes(monkeypatch):
    monkeypatch.setattr("robolineage_shared_agents.llm_routes._load_dotenv_if_available", lambda: None)
    for name in (
        "ROBOLINEAGE_DISCOVERY_LLM_API_KEY",
        "ROBOLINEAGE_DISCOVERY_LLM_MODEL",
        "ROBOLINEAGE_DISCOVERY_LLM_BASE_URL",
        "ROBOLINEAGE_AGENT_API_KEY",
        "ROBOLINEAGE_AGENT_MODEL",
        "ROBOLINEAGE_AGENT_BASE_URL",
        "TASK_LLM_API_KEY",
        "TASK_LLM_MODEL",
        "TASK_LLM_BASE_URL",
        "VSA_VLM_API_KEY",
        "VSA_VLM_MODEL",
        "VSA_VLM_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VSA_VLM_API_KEY", "vsa-key")
    monkeypatch.setenv("VSA_VLM_MODEL", "vsa-model")
    monkeypatch.setenv("VSA_VLM_BASE_URL", "https://vsa.example/v1")
    monkeypatch.setenv("TASK_LLM_API_KEY", "task-key")
    monkeypatch.setenv("TASK_LLM_MODEL", "task-model")
    monkeypatch.setenv("TASK_LLM_BASE_URL", "https://task.example/v1")
    monkeypatch.setenv("ROBOLINEAGE_DISCOVERY_LLM_API_KEY", "discovery-key")
    monkeypatch.setenv("ROBOLINEAGE_DISCOVERY_LLM_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setenv("ROBOLINEAGE_DISCOVERY_LLM_BASE_URL", "https://gateway.example/v1")

    client = OpenAICompatibleDiscoveryClient.from_env()

    assert client is not None
    assert client.api_key == "discovery-key"
    assert client.model == "anthropic/claude-sonnet-4.6"
    assert client.base_url == "https://gateway.example/v1"


def test_discovery_client_defaults_to_sonnet_for_openai_compatible_route(monkeypatch):
    monkeypatch.setattr("robolineage_shared_agents.llm_routes._load_dotenv_if_available", lambda: None)
    for name in (
        "ROBOLINEAGE_DISCOVERY_LLM_API_KEY",
        "ROBOLINEAGE_DISCOVERY_LLM_MODEL",
        "ROBOLINEAGE_DISCOVERY_LLM_BASE_URL",
        "ROBOLINEAGE_AGENT_API_KEY",
        "ROBOLINEAGE_AGENT_MODEL",
        "ROBOLINEAGE_AGENT_BASE_URL",
        "TASK_LLM_API_KEY",
        "TASK_LLM_MODEL",
        "TASK_LLM_BASE_URL",
        "VSA_VLM_API_KEY",
        "VSA_VLM_MODEL",
        "VSA_VLM_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROBOLINEAGE_DISCOVERY_LLM_API_KEY", "discovery-key")
    monkeypatch.setenv("ROBOLINEAGE_DISCOVERY_LLM_BASE_URL", "https://gateway.example/v1")

    client = OpenAICompatibleDiscoveryClient.from_env()

    assert client is not None
    assert client.model == "anthropic/claude-sonnet-4.6"


def test_discovery_client_uses_task_llm_route_when_discovery_route_is_absent(monkeypatch):
    monkeypatch.setattr("robolineage_shared_agents.llm_routes._load_dotenv_if_available", lambda: None)
    for name in (
        "ROBOLINEAGE_DISCOVERY_LLM_API_KEY",
        "ROBOLINEAGE_DISCOVERY_LLM_MODEL",
        "ROBOLINEAGE_DISCOVERY_LLM_BASE_URL",
        "ROBOLINEAGE_AGENT_API_KEY",
        "ROBOLINEAGE_AGENT_MODEL",
        "ROBOLINEAGE_AGENT_BASE_URL",
        "TASK_LLM_API_KEY",
        "TASK_LLM_MODEL",
        "TASK_LLM_BASE_URL",
        "VSA_VLM_API_KEY",
        "VSA_VLM_MODEL",
        "VSA_VLM_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VSA_VLM_API_KEY", "vsa-key")
    monkeypatch.setenv("VSA_VLM_MODEL", "vsa-model")
    monkeypatch.setenv("VSA_VLM_BASE_URL", "https://vsa.example/v1")
    monkeypatch.setenv("TASK_LLM_API_KEY", "task-key")
    monkeypatch.setenv("TASK_LLM_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setenv("TASK_LLM_BASE_URL", "https://task.example/v1")

    client = OpenAICompatibleDiscoveryClient.from_env()

    assert client is not None
    assert client.api_key == "task-key"
    assert client.model == "anthropic/claude-sonnet-4.6"
    assert client.base_url == "https://task.example/v1"


def test_discovery_client_does_not_use_vsa_route_as_fallback(monkeypatch):
    monkeypatch.setattr("robolineage_shared_agents.llm_routes._load_dotenv_if_available", lambda: None)
    for name in (
        "ROBOLINEAGE_DISCOVERY_LLM_API_KEY",
        "ROBOLINEAGE_DISCOVERY_LLM_MODEL",
        "ROBOLINEAGE_DISCOVERY_LLM_BASE_URL",
        "ROBOLINEAGE_AGENT_API_KEY",
        "ROBOLINEAGE_AGENT_MODEL",
        "ROBOLINEAGE_AGENT_BASE_URL",
        "TASK_LLM_API_KEY",
        "TASK_LLM_MODEL",
        "TASK_LLM_BASE_URL",
        "VSA_VLM_API_KEY",
        "VSA_VLM_MODEL",
        "VSA_VLM_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VSA_VLM_API_KEY", "vsa-key")
    monkeypatch.setenv("VSA_VLM_MODEL", "vsa-model")
    monkeypatch.setenv("VSA_VLM_BASE_URL", "https://vsa.example/v1")

    assert OpenAICompatibleDiscoveryClient.from_env() is None


def test_framework_discovery_does_not_autowire_legacy_video_pose_adapter(tmp_path):
    repo = tmp_path / "video_pose_repo"
    (repo / "utils").mkdir(parents=True)
    (repo / "train.py").write_text("print('train')\n", encoding="utf-8")
    (repo / "utils" / "utils.py").write_text(
        "export_path = 'export.json'\n"
        "pose_path = 'pose.h5'\n"
        "frames_path = 'frames.csv'\n"
        "video_dir = 'videos'\n",
        encoding="utf-8",
    )

    launcher = tmp_path / "terminal.py"
    launcher.write_text("print('terminal launcher')\n", encoding="utf-8")
    discovery = FrameworkDiscoveryAgent().discover(
        repo_root=repo,
        output_dir=tmp_path / "profiles" / "video_pose",
        framework_type="custom_policy",
        commands=CommandIntake(
            train_command=f"{sys.executable} train.py --datasets {{dataset_output}} --ckpt_dir {{checkpoint_dir}}",
        ),
        train_launch_mode="external_terminal",
        terminal_command=(sys.executable, str(launcher), "{script}"),
    )

    profile = load_framework_profile(discovery.profile_path)
    assert profile.framework_type == "custom_policy"
    assert profile.staging.selected_rollouts_dir is None
    assert profile.staging.framework_input_dir is None
    assert profile.dataset_command is None
    assert profile.execution.train_launch_mode == "external_terminal"
    assert profile.execution.terminal_command == (sys.executable, str(launcher), "{script}")
    assert profile.dataset_adapter["adapter_id"] == "missing_dataset_converter"
    assert profile.dataset_adapter["strategy"] == "requires_dataset_command"

    discovery_payload = json.loads(discovery.discovery_path.read_text(encoding="utf-8"))
    assert discovery_payload["execution"]["train_launch_mode"] == "external_terminal"
    assert discovery_payload["dataset_adapter"]["adapter_id"] == "missing_dataset_converter"
    assert discovery_payload["dataset_adapter"]["strategy"] == "requires_dataset_command"
    assert discovery_payload["staging"] == {
        "selected_rollouts_file": "{staging_dir}/selected_rollouts.json",
    }
    assert "commands_do_not_reference_ROBOLINEAGE_inputs" not in discovery_payload["warnings"]


def test_discovery_generates_repo_specific_hdf5_adapter_and_monitor(tmp_path):
    h5py = pytest.importorskip("h5py")

    rollouts = tmp_path / "rollouts"
    rollout = _rollout(rollouts, "r1", decision="accepted")
    _write_raw_rosbag_rollout(rollout / "raw")

    repo = tmp_path / "transengram_like"
    (repo / "tools").mkdir(parents=True)
    (repo / "act").mkdir(parents=True)
    (repo / "README.md").write_text("# Transengram-like ACT repo\n", encoding="utf-8")
    (repo / "act" / "train.py").write_text(
        "\n".join(
            [
                'WEB_PROGRESS_PREFIX = "WEBCTRL_JSON "',
                "def emit_web_progress(event, **payload): pass",
                "emit_web_progress('dataset_scanned')",
                "emit_web_progress('epoch_completed')",
                "emit_web_progress('train_finished')",
                "ckpt_name = 'policy_best.ckpt'",
            ]
        ),
        encoding="utf-8",
    )
    train_script = repo / "tools" / "02_train.sh"
    train_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"{json.dumps(sys.executable)} - <<'PY'",
                "import json, os",
                "from pathlib import Path",
                "dataset = Path(os.environ['DATASET_DIR'])",
                "ckpt = Path(os.environ['CKPT_DIR']); ckpt.mkdir(parents=True, exist_ok=True)",
                "episodes = sorted(dataset.glob('episode_*.hdf5'))",
                "print('WEBCTRL_JSON ' + json.dumps({'event': 'dataset_scanned', 'num_episodes': len(episodes)}))",
                "print('WEBCTRL_JSON ' + json.dumps({'event': 'epoch_completed', 'epoch': 1, 'train_loss': 0.4, 'val_loss': 0.3}))",
                "(ckpt / 'policy_best.ckpt').write_text('policy')",
                "print('WEBCTRL_JSON ' + json.dumps({'event': 'train_finished', 'best_epoch': 1, 'best_val_loss': 0.3}))",
                "PY",
            ]
        ),
        encoding="utf-8",
    )
    train_script.chmod(0o755)

    discovery = FrameworkDiscoveryAgent().discover(
        repo_root=repo,
        output_dir=tmp_path / "profiles" / "act_hdf5",
        target_dataset_format=(
            "episode_X.hdf5\n"
            "├── attrs['sim']\n"
            "├── action shape (T, 14)\n"
            "├── action_eef shape (T, 14)\n"
            "└── observations/\n"
            "    ├── qpos shape (T, 14)\n"
            "    ├── qvel shape (T, 14)\n"
            "    └── images/\n"
            "        ├── head\n"
            "        ├── left_wrist\n"
            "        └── right_wrist"
        ),
        command_context=(
            f"repo_root: {repo}\n"
            "conda env: act\n"
            "train command: CAMERA_NAMES='head right_wrist' bash tools/02_train.sh"
        ),
        commands=CommandIntake(train_command="bash tools/02_train.sh"),
    )
    profile = load_framework_profile(discovery.profile_path)

    assert profile.dataset_command is not None
    assert profile.dataset_adapter["strategy"] == "registered_adapter_module"
    assert profile.dataset_adapter["module"] == "robolineage_train.dataset_adapters.rosbag_act_hdf5"
    assert "--target-hz" in profile.dataset_command.args
    assert "30" in profile.dataset_command.args
    assert profile.train_command is not None
    assert "DATASET_DIR={dataset_output}" in profile.train_command.args[2]
    assert profile.outputs.checkpoint_glob == "{checkpoint_dir}/policy_best.ckpt"
    assert profile.monitor["json_line_prefixes"] == ["WEBCTRL_JSON "]
    assert profile.monitor["events"] == ["dataset_scanned", "epoch_completed", "train_finished"]
    assert profile.dataset_adapter["target_contract"]["dataset_kind"] == "act_hdf5"

    result = FrameworkAdapter(profile).run(
        rollouts_root=rollouts,
        workspace_dir=tmp_path / "ROBOLINEAGE_run",
        dataset_version="v1",
        policy_version="1.0.0",
    )

    episode = result.dataset_output_dir / "episode_0.hdf5"
    assert episode.exists()
    with h5py.File(episode, "r") as root:
        assert root.attrs["frame_rate"] == 30
        assert root["/action"].shape == (10, 14)
        assert root["/observations/qpos"].shape == (10, 14)
        qpos = root["/observations/qpos"][()]
        assert qpos[:, 0].tolist() == pytest.approx([i / 9.0 for i in range(10)], abs=1e-5)
        assert qpos[:, 7].tolist() == pytest.approx([10.0 + (2.0 * i / 9.0) for i in range(10)], abs=1e-5)
        assert set(root["/observations/images"].keys()) == {"head", "right_wrist"}
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.name == "policy_best.ckpt"
    status = json.loads(result.training_status_path.read_text(encoding="utf-8"))
    assert status["metrics"]["event"] == "train_finished"
    assert status["metrics"]["epoch"] == 1
    assert status["metrics"]["loss"] == 0.3
    adapt_status = json.loads(result.dataset_adapt_status_path.read_text(encoding="utf-8"))
    assert adapt_status["status"] == "completed"
    assert adapt_status["target_dataset_kind"] == "act_hdf5"
    assert adapt_status["adapter_report"]["exported_episode_count"] == 1
    discovery_payload = json.loads(discovery.discovery_path.read_text(encoding="utf-8"))
    assert discovery_payload["dataset_adapter"]["adapter_id"] == "rosbag_act_hdf5"
    assert discovery_payload["target_contract"]["dataset_kind"] == "act_hdf5"
    assert [item["event"] for item in discovery_payload["events"]][-1] == "discovery_completed"
    assert discovery_payload["monitor"]["strategy"] == "generated_from_repo_training_log_code"
    assert discovery_payload["integration_manifest"]["schema_version"] == "RoboLineage.training_integration_manifest.v1"
    manifest_outputs = discovery_payload["integration_manifest"]["outputs"]
    assert manifest_outputs["stdout_capture"] == "{workspace_dir}/train_command.log"
    assert manifest_outputs["checkpoint_glob"] == "{checkpoint_dir}/policy_best.ckpt"


def test_framework_profile_loads_tmux_execution(tmp_path):
    profile_path = tmp_path / "framework.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "name: tmux_framework",
                "repo_root: .",
                "execution:",
                "  train_launch_mode: tmux",
                "  tmux_session_name: ROBOLINEAGE_custom_{policy_version}",
                "train_command:",
                "  args: [python, train.py]",
            ]
        ),
        encoding="utf-8",
    )

    profile = load_framework_profile(profile_path)

    assert profile.execution.train_launch_mode == "tmux"
    assert profile.execution.tmux_session_name == "ROBOLINEAGE_custom_{policy_version}"


def test_framework_adapter_writes_failed_status_when_host_command_fails(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "a", decision="accepted")
    repo = tmp_path / "host_repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "fail.py").write_text(
        "import sys; print('step=1'); sys.exit(7)",
        encoding="utf-8",
    )
    profile_path = tmp_path / "framework.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "name: failing_framework",
                "repo_root: host_repo",
                "train_command:",
                f"  args: [{json.dumps(sys.executable)}, scripts/fail.py]",
                "log_patterns:",
                "  step: 'step=(\\d+)'",
            ]
        ),
        encoding="utf-8",
    )
    profile = load_framework_profile(profile_path)
    workspace = tmp_path / "ROBOLINEAGE_run"

    try:
        FrameworkAdapter(profile).run(
            rollouts_root=rollouts,
            workspace_dir=workspace,
            dataset_version="v1",
            policy_version="1.0.0",
        )
    except Exception as exc:
        assert getattr(exc, "returncode", None) == 7
    else:
        raise AssertionError("expected host command failure")

    status = json.loads((workspace / "training_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["returncode"] == 7
    assert (workspace / "train_command.log").read_text(encoding="utf-8").strip() == "step=1"


def test_framework_adapter_marks_zero_exit_unstable_from_training_log(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "a", decision="accepted")
    repo = tmp_path / "host_repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "train.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)",
                "(out / 'policy.ckpt').write_text('checkpoint')",
                "print('step=1 loss=0.2')",
                "print('step=2 loss=nan')",
            ]
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "framework.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "name: unstable_framework",
                f"repo_root: {repo}",
                "train_command:",
                f"  args: [{json.dumps(sys.executable)}, scripts/train.py, '{{checkpoint_dir}}']",
                "outputs:",
                "  checkpoint_glob: '{checkpoint_dir}/*.ckpt'",
            ]
        ),
        encoding="utf-8",
    )
    profile = load_framework_profile(profile_path)
    result = FrameworkAdapter(profile).run(
        rollouts_root=rollouts,
        workspace_dir=tmp_path / "ROBOLINEAGE_run",
        dataset_version="v1",
        policy_version="1.0.0",
    )

    status = json.loads(result.training_status_path.read_text(encoding="utf-8"))
    assert status["status"] == "unstable"
    assert status["metrics"]["recommended_action"] == "inspect_training_instability"


def test_framework_adapter_can_launch_train_in_external_terminal_mode(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "a", decision="accepted")
    repo = tmp_path / "host_repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "train.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)",
                "ckpt = out / 'policy.ckpt'; ckpt.write_text('checkpoint')",
                "print('step=9 loss=0.19')",
                "print('saved checkpoint to ' + str(ckpt))",
            ]
        ),
        encoding="utf-8",
    )
    launcher = tmp_path / "fake_terminal.py"
    launcher.write_text(
        "import subprocess, sys\nraise SystemExit(subprocess.call(['bash', sys.argv[1]]))\n",
        encoding="utf-8",
    )
    profile_path = tmp_path / "framework.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "name: external_terminal_framework",
                f"repo_root: {repo}",
                "train_command:",
                f"  args: [{json.dumps(sys.executable)}, scripts/train.py, '{{checkpoint_dir}}']",
                "outputs:",
                "  checkpoint_glob: '{checkpoint_dir}/*.ckpt'",
                "execution:",
                "  train_launch_mode: external_terminal",
                "  terminal_hold_open: false",
                "  terminal_command:",
                f"    - {json.dumps(sys.executable)}",
                f"    - {json.dumps(str(launcher))}",
                "    - '{script}'",
            ]
        ),
        encoding="utf-8",
    )

    result = FrameworkAdapter(load_framework_profile(profile_path)).run(
        rollouts_root=rollouts,
        workspace_dir=tmp_path / "ROBOLINEAGE_run",
        dataset_version="v1",
        policy_version="1.0.0",
    )

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.name == "policy.ckpt"
    status = json.loads(result.training_status_path.read_text(encoding="utf-8"))
    assert status["metrics"]["launch_mode"] == "external_terminal"
    assert status["metrics"]["step"] == 9


def test_parse_training_log_supports_jsonl_and_regex():
    parsed = parse_training_log(
        "\n".join([
            '{"step": 3, "loss": 0.9}',
            "success_rate=0.75",
        ]),
        {"success_rate": "success_rate=([0-9.]+)"},
    )

    assert parsed["step"] == 3
    assert parsed["loss"] == 0.9
    assert parsed["success_rate"] == 0.75
    assert parsed["status"] == "completed"


def test_parse_training_log_supports_generated_json_prefix_monitor():
    parsed = parse_training_log(
        "\n".join([
            'WEBCTRL_JSON {"event": "epoch_completed", "epoch": 2, "train_loss": 0.42, "val_loss": 0.31}',
            'WEBCTRL_JSON {"event": "train_finished", "best_epoch": 2, "best_val_loss": 0.31}',
        ]),
        {},
        {"json_line_prefixes": ["WEBCTRL_JSON "]},
    )

    assert parsed["event"] == "train_finished"
    assert parsed["epoch"] == 2
    assert parsed["loss"] == 0.31
    assert parsed["best_val_loss"] == 0.31
