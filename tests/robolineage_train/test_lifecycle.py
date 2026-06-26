from __future__ import annotations

import json
import sys
from pathlib import Path

from robolineage_train import TrainingLifecycleRunner, load_framework_profile
from robolineage_train.__main__ import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rollout(root: Path, rollout_id: str) -> None:
    path = root / rollout_id
    path.mkdir(parents=True)
    _write_json(
        path / "dataset_admission.json",
        {
            "decision": "accepted",
            "data_use": ["success_trajectory", "recovery_training"],
            "reasons": ["final_success_with_recovered_failure"],
            "task_description": "stack block",
        },
    )
    _write_json(
        path / "rollout_summary.json",
        {
            "rollout_id": rollout_id,
            "final_success": True,
            "success_confidence": 0.82,
            "task_description": "stack block",
        },
    )
    _write_json(
        path / "annotation.final.json",
        {
            "task": {"phases": ["approach", "grasp", "place"]},
            "l1_annotation": {"phases": ["approach", "grasp", "place"]},
        },
    )
    _write_json(
        path / "failure_analysis.json",
        {
            "candidate_count": 1,
            "failure_events": [{"failure_type": "grasp_miss", "recovered": True}],
        },
    )


def _uncertain_trainable_rollout(root: Path, rollout_id: str) -> None:
    path = root / rollout_id
    path.mkdir(parents=True)
    _write_json(
        path / "dataset_admission.json",
        {
            "decision": "needs_review",
            "accepted_for_training": True,
            "label_quality": "uncertain",
            "review_reason": "phase labels need correction but trajectory was human supervised",
            "data_use": ["success_trajectory", "review_queue"],
            "reasons": ["label_quality_uncertain"],
            "task_description": "stack block",
        },
    )
    _write_json(
        path / "rollout_summary.json",
        {
            "rollout_id": rollout_id,
            "final_success": False,
            "success_confidence": 0.35,
            "task_description": "stack block",
        },
    )
    _write_json(
        path / "annotation.final.json",
        {
            "task": {"phases": ["approach", "grasp", "place"]},
            "l1_annotation": {"phases": ["approach", "grasp", "place"]},
        },
    )
    _write_json(
        path / "failure_analysis.json",
        {
            "candidate_count": 0,
            "failure_events": [],
        },
    )


def _host_repo(repo: Path) -> None:
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "build_dataset.py").write_text(
        "import sys, json\n"
        "from pathlib import Path\n"
        "selected=json.loads(Path(sys.argv[1]).read_text())\n"
        "out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'dataset.json').write_text(json.dumps(selected))\n"
        "print('dataset_count=' + str(selected['selected_rollout_count']))\n",
        encoding="utf-8",
    )
    (scripts / "train.py").write_text(
        "import sys, json\n"
        "from pathlib import Path\n"
        "out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)\n"
        "ckpt=out/'policy.ckpt'; ckpt.write_text('policy')\n"
        "(out/'training.log').write_text('\\n'.join([\n"
        "  json.dumps({'step': 1, 'loss': 1.0}),\n"
        "  json.dumps({'step': 20, 'loss': 0.2, 'checkpoint': str(ckpt)}),\n"
        "]))\n"
        "print(json.dumps({'step': 20, 'loss': 0.2, 'checkpoint': str(ckpt)}))\n",
        encoding="utf-8",
    )
    (scripts / "eval.py").write_text(
        "import sys, json\n"
        "from pathlib import Path\n"
        "out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'result.json').write_text(json.dumps({'success_rate': 0.86}))\n"
        "print(json.dumps({'success_rate': 0.86}))\n",
        encoding="utf-8",
    )


def _profile(path: Path, repo: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "name: realistic_generic_framework",
                "framework_type: generic_policy",
                f"repo_root: {repo}",
                "dataset_command:",
                f"  args: [{json.dumps(sys.executable)}, scripts/build_dataset.py, '{{selected_rollouts_file}}', '{{dataset_output}}']",
                "train_command:",
                f"  args: [{json.dumps(sys.executable)}, scripts/train.py, '{{dataset_output}}', '{{checkpoint_dir}}']",
                "eval_command:",
                f"  args: [{json.dumps(sys.executable)}, scripts/eval.py, '{{checkpoint_path}}', '{{eval_output}}']",
                "outputs:",
                "  checkpoint_glob: '{checkpoint_dir}/*.ckpt'",
                "  train_log: '{checkpoint_dir}/training.log'",
                "  eval_result: '{eval_output}/result.json'",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_training_lifecycle_writes_dataset_policy_context_and_recommendation(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "r1")
    repo = tmp_path / "host_repo"
    _host_repo(repo)
    profile = load_framework_profile(_profile(tmp_path / "framework.yaml", repo))

    result = TrainingLifecycleRunner(
        profile=profile,
        rollouts_root=rollouts,
        datasets_root=tmp_path / "datasets",
        workspace_root=tmp_path / "runs",
    ).run(policy_version="1.0.0")

    assert result.dataset_version == "v1"
    assert result.dataset_lock_path.exists()
    meta = json.loads(result.policy_meta_path.read_text(encoding="utf-8"))
    assert meta["trained_on_dataset"] == "v1"
    assert meta["framework_name"] == "realistic_generic_framework"
    assert meta["eval_success_rate"] == 0.86
    recommendation = json.loads(result.deployment_recommendation_path.read_text(encoding="utf-8"))
    assert recommendation["decision"] == "deploy_recommended"
    context = json.loads(result.ROBOLINEAGE_context_path.read_text(encoding="utf-8"))
    assert context["post_review_summary"]["failure_type_counts"] == {"grasp_miss": 1}
    assert (result.workspace_dir / "dataset_health_report.json").exists()
    assert (result.workspace_dir / "dataset_health_understanding.json").exists()
    assert context["dataset"]["dataset_health_report_path"] == str(result.workspace_dir / "dataset_health_report.json")


def test_lifecycle_cli(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "r1")
    repo = tmp_path / "host_repo"
    _host_repo(repo)
    profile = _profile(tmp_path / "framework.yaml", repo)

    assert main([
        "lifecycle-run",
        "--profile", str(profile),
        "--rollouts-root", str(rollouts),
        "--datasets-root", str(tmp_path / "datasets"),
        "--workspace-root", str(tmp_path / "runs"),
        "--policy-version", "1.0.0",
    ]) == 0

    assert (tmp_path / "datasets" / "v1" / "dataset.lock").exists()


def test_training_lifecycle_auto_continues_dataset_versions(tmp_path):
    rollouts = tmp_path / "rollouts"
    _rollout(rollouts, "r1")
    repo = tmp_path / "host_repo"
    _host_repo(repo)
    profile = load_framework_profile(_profile(tmp_path / "framework.yaml", repo))

    first = TrainingLifecycleRunner(
        profile=profile,
        rollouts_root=rollouts,
        datasets_root=tmp_path / "datasets",
        workspace_root=tmp_path / "runs",
    ).run(policy_version="1.0.0")
    second = TrainingLifecycleRunner(
        profile=profile,
        rollouts_root=rollouts,
        datasets_root=tmp_path / "datasets",
        workspace_root=tmp_path / "runs",
    ).run(policy_version="1.0.1")

    assert first.dataset_version == "v1"
    assert second.dataset_version == "v2"
    assert (tmp_path / "datasets" / "v2" / "dataset.lock").exists()
    assert first.workspace_dir != second.workspace_dir


def test_train_manifest_includes_rollouts_accepted_for_training_even_when_labels_need_review(tmp_path):
    from robolineage_train.lifecycle import build_train_manifest_from_post_review

    rollouts = tmp_path / "rollouts"
    _uncertain_trainable_rollout(rollouts, "r_uncertain")

    entries = build_train_manifest_from_post_review(
        rollouts_root=rollouts,
        output_path=tmp_path / "train_manifest.jsonl",
    )

    assert [entry.rollout_id for entry in entries] == ["r_uncertain"]
    assert entries[0].review_score == "B"
    rows = [
        json.loads(line)
        for line in (tmp_path / "train_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["rollout_id"] == "r_uncertain"
