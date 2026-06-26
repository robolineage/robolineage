from __future__ import annotations

from robolineage_train import DatasetAdapterAgent


def test_dataset_adapter_agent_rejects_legacy_video_pose_autowire(tmp_path):
    repo = tmp_path / "repo"
    utils = repo / "utils"
    utils.mkdir(parents=True)
    (repo / "train.py").write_text("print('train')\n", encoding="utf-8")
    (utils / "utils.py").write_text(
        "\n".join(
            [
                "export_path = 'export.json'",
                "pose_path = 'pose.h5'",
                "frames_path = 'frames.csv'",
                "video_dir = 'videos'",
            ]
        ),
        encoding="utf-8",
    )

    plan = DatasetAdapterAgent().plan(
        repo_root=repo,
        framework_type="custom_policy",
        train_command="python train.py --datasets {dataset_output}",
    )

    assert plan.strategy == "requires_dataset_command"
    assert plan.dataset_command is None
    assert plan.source_data_policy == "read_only"
    assert any("dataset conversion strategy" in item for item in plan.warnings)


def test_dataset_adapter_agent_uses_user_dataset_command_as_authority(tmp_path):
    plan = DatasetAdapterAgent().plan(
        repo_root=tmp_path,
        dataset_command="python convert.py --in {selected_rollouts_file} --out {dataset_output}",
        train_command="python train.py --data {dataset_output}",
    )

    assert plan.strategy == "user_supplied"
    assert plan.confidence == 1.0
    assert plan.dataset_command == (
        "python",
        "convert.py",
        "--in",
        "{selected_rollouts_file}",
        "--out",
        "{dataset_output}",
    )


def test_dataset_adapter_agent_keeps_direct_manifest_training_without_conversion(tmp_path):
    plan = DatasetAdapterAgent().plan(
        repo_root=tmp_path,
        train_command="python train.py --manifest {selected_rollouts_file}",
    )

    assert plan.strategy == "direct_manifest"
    assert plan.dataset_command is None
    assert plan.train_input == "{selected_rollouts_file}"
    assert plan.output_path == "{selected_rollouts_file}"


def test_dataset_adapter_agent_warns_when_dataset_output_needs_converter(tmp_path):
    repo = tmp_path / "unknown_repo"
    repo.mkdir()

    plan = DatasetAdapterAgent().plan(
        repo_root=repo,
        train_command="python train.py --dataset {dataset_output}",
    )

    assert plan.strategy == "requires_dataset_command"
    assert plan.dataset_command is None
    assert any("dataset conversion strategy" in item for item in plan.warnings)
