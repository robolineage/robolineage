from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


REMOVED_LEGACY_PATHS = (
    "src/legacy_agents/success_risk",
    "src/legacy_agents/strategy",
    "src/legacy_agents/health",
    "src/legacy_agents/master_review",
    "src/legacy_agents",
    "src/legacy_agents.egg-info",
    "src/legacy_l1",
    "src/legacy_p_rollout",
    "src/robolineage_post_rollout/reviewer.py",
    "src/robolineage_contracts/agents/health.py",
    "src/robolineage_contracts/agents/master.py",
    "src/robolineage_contracts/agents/strategy.py",
    "src/robolineage_contracts/agents/success_risk.py",
    "src/robolineage_contracts/pipeline/report.py",
    "src/robolineage_contracts/pipeline/trajectory.py",
    "src/legacy_t1_agents",
    "src/robolineage_contracts/stream",
    "src/robolineage_schemas/frame_msg.schema.json",
    "src/robolineage_schemas/action_msg.schema.json",
    "tests/agents",
    "tests/legacy_agents",
    "tests/legacy_agents_master",
    "tests/legacy_l1",
    "tests/p_rollout",
    "tests/legacy_t1_agents",
    "tests/robolineage_contracts/stream",
    "tests/robolineage_contracts/agents/test_health.py",
    "tests/robolineage_contracts/agents/test_master.py",
    "tests/robolineage_contracts/agents/test_strategy.py",
    "tests/robolineage_contracts/agents/test_success_risk.py",
    "tests/robolineage_contracts/pipeline/test_report.py",
    "tests/robolineage_contracts/pipeline/test_trajectory.py",
    "tests/robolineage_data_source/bus",
    "tests/_shared_fixtures/p_rollout_report.json",
    "docs/superpowers/plans",
    "docs/superpowers/specs",
    "doc/archive",
    "docs/visual_snapshot_agent_design",
    "scripts/bootstrap_integration_trunk.sh",
    "scripts/run_master_review.py",
    "scripts/data_source/rollout_writer_cli.py",
    "scripts/vsa_offline.py",
    "src/robolineage_data_source/adapters/hdf5_replay.py",
    "src/robolineage_data_source/recorder",
    "src/robolineage_shared_agents/visual_snapshot/archive_loader.py",
    "src/robolineage_shared_agents/visual_snapshot/action_event_detector.py",
    "src/robolineage_shared_agents/visual_snapshot/action_signal_builder.py",
    "src/robolineage_shared_agents/visual_snapshot/action_window_builder.py",
    "src/robolineage_shared_agents/visual_snapshot/pipeline.py",
    "src/robolineage_shared_agents/visual_snapshot/sequence_loader.py",
    "src/robolineage_shared_agents/visual_snapshot/window_builder.py",
    "src/robolineage_train/dataset_adapters/video_pose_export.py",
    "src/robolineage_contracts/core/frame.py",
    "src/robolineage_contracts/core/pose.py",
    "tests/robolineage_data_source/adapters/test_hdf5_replay.py",
    "tests/robolineage_data_source/recorder",
    "tests/robolineage_data_source/test_rollout_writer_cli.py",
    "tests/robolineage_contracts/core/test_frame.py",
    "tests/robolineage_contracts/core/test_pose.py",
    "tests/robolineage_train/test_video_pose_dataset_adapter.py",
    "tests/visual_snapshot/test_offline_migration.py",
    "tests/visual_snapshot/test_vsa_offline_cli.py",
    "tests/visual_snapshot/fixtures/action_rollout",
    "tests/_shared_fixtures/generate.py",
    "tests/_shared_fixtures/mini_rollout/frames.csv",
    "doc/schemas/examples/frames.example.csv",
    ".tmp/pytest-linux-full.log",
    ".tmp/pytest-linux-py310-full.log",
)


FORBIDDEN_CURRENT_DOC_REFERENCES = (
    "docs/superpowers/plans",
    "docs/superpowers/plans/",
    "docs/superpowers/specs",
    "docs/superpowers/specs/",
    "doc/archive",
    "doc/archive/",
    "docs/visual_snapshot_agent_design",
    "src/legacy_agents/{success_risk,strategy,health,master_review}",
    "src/legacy_agents",
    "src/legacy_agents/",
    "src/legacy_agents.egg-info",
    "legacy-agents",
    "tests/legacy_agents",
    "tests/legacy_agents/",
    "tests/legacy_agents_master",
    "tests/legacy_agents_master/",
    "from legacy_agents",
    "import legacy_agents",
    "legacy_agents.",
    "src/legacy_agents/success_risk/",
    "src/legacy_agents/strategy/",
    "src/legacy_agents/health/",
    "src/legacy_agents/master_review/",
    "src/legacy_l1",
    "src/legacy_l1/",
    "tests/legacy_l1",
    "tests/legacy_l1/",
    "legacy_l1",
    "agents,l1,dataset",
    "l1 = []",
    "legacy_p_rollout",
    "src/legacy_p_rollout/",
    "legacy_t1_agents",
    "src/legacy_t1_agents/",
    "src/robolineage_contracts/stream",
    "src/robolineage_contracts/stream/",
    "robolineage_contracts.stream",
    "frame_msg.schema.json",
    "action_msg.schema.json",
    "tests/agents/",
    "tests/p_rollout/",
    "tests/legacy_t1_agents/",
    "tests/robolineage_contracts/stream",
    "tests/robolineage_contracts/stream/",
    "src/robolineage_contracts/agents/strategy.py",
    "src/robolineage_contracts/pipeline/trajectory.py",
    "src/robolineage_contracts/pipeline/report.py",
    "tests/robolineage_data_source/bus",
    "tests/robolineage_data_source/bus/",
    "robolineage_data_source.bus",
    "scripts/bootstrap_integration_trunk.sh",
    "scripts/run_master_review.py",
    "scripts/data_source/rollout_writer_cli.py",
    "scripts/vsa_offline.py",
    "hdf5_replay.py",
    "rollout_dir_writer.py",
    "pose_writer.py",
    "video_encoder.py",
    "video_encoder",
    "pipeline.run_action_guided_rollout",
    "run_action_guided_rollout",
    "video_pose_export",
    "video_pose_export_package",
)


CURRENT_DOCS = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "run.sh",
    ROOT / "configs" / "README.md",
    ROOT / "configs" / "robolineage_default.yaml",
    ROOT / "configs" / "arx_one.yaml",
    ROOT / "configs" / "arx_one_rs3.yaml",
    ROOT / "docs" / "overview.md",
    ROOT / "docs" / "artifact_contracts.md",
    ROOT / "docs" / "operator_workflow.md",
    ROOT / "docs" / "prompt_contracts.md",
    ROOT / "docs" / "reproducibility.md",
    ROOT / "docs" / "deployment" / "README.md",
    ROOT / "src" / "robolineage_contracts" / "README.md",
    ROOT / "src" / "robolineage_train" / "README.md",
    ROOT / "tests" / "_shared_fixtures" / "README.md",
    ROOT / "tests" / "robolineage_contracts" / "README.md",
    ROOT / "scripts" / "README.md",
    ROOT / "pyproject.toml",
    ROOT / "scripts" / "OWNERSHIP.yaml",
    ROOT / "scripts" / "check_contracts_imports.py",
)


FORBIDDEN_CURRENT_CODE_REFERENCES = (
    "act_legacy_hdf5",
    "generated_act_legacy_hdf5",
    "p_rollout_report",
    "PRolloutReport",
    "SuccessRiskReport",
    "CollectionStrategyRecommendation",
    "CollectionConfig",
    "MasterReviewVerdict",
    "DatasetHealthReport",
    "TrajectorySegment",
    "FailureAttribution",
    "RecoverySegment",
    "QualityScore",
    "robolineage_contracts.agents.strategy",
    "robolineage_contracts.pipeline.trajectory",
    "robolineage_contracts.pipeline.report",
    "HDF5ReplayAdapter",
    "robolineage_data_source.adapters.hdf5_replay",
    "RolloutDirWriter",
    "PoseWriter",
    "VideoEncoder",
    "video_encoder",
    "run_action_guided_rollout",
    "SequenceLoader",
    "ActionWindowBuilder",
    "robolineage_train.dataset_adapters.video_pose_export",
    "video_pose_export_package",
)


CURRENT_CODE_ROOTS = (
    ROOT / "src" / "robolineage_app",
    ROOT / "src" / "robolineage_data_source",
    ROOT / "src" / "robolineage_shared_agents" / "visual_snapshot",
    ROOT / "src" / "robolineage_train",
    ROOT / "tests" / "robolineage_app",
    ROOT / "tests" / "robolineage_data_source",
    ROOT / "tests" / "visual_snapshot",
    ROOT / "tests" / "robolineage_train",
    ROOT / "src" / "robolineage_contracts",
    ROOT / "tests" / "robolineage_contracts",
)


def test_legacy_runtime_and_plan_artifacts_are_removed() -> None:
    leftovers = [path for path in REMOVED_LEGACY_PATHS if (ROOT / path).exists()]

    assert leftovers == []


def test_current_entrypoints_do_not_link_deleted_legacy_paths() -> None:
    offenders: list[str] = []

    for path in CURRENT_DOCS:
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN_CURRENT_DOC_REFERENCES:
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)} -> {needle}")

    assert offenders == []


def test_current_code_uses_current_act_hdf5_names() -> None:
    offenders: list[str] = []

    for root in CURRENT_CODE_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".md"}:
                continue
            if path.name == "CHANGELOG.md":
                continue
            text = path.read_text(encoding="utf-8")
            for needle in FORBIDDEN_CURRENT_CODE_REFERENCES:
                if needle in text:
                    offenders.append(f"{path.relative_to(ROOT)} -> {needle}")

    assert offenders == []


def test_fastapi_apps_use_lifespan_not_on_event() -> None:
    offenders: list[str] = []

    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "@app.on_event(" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_console_startup_does_not_create_business_task_dir() -> None:
    run_sh = (ROOT / "run.sh").read_text(encoding="utf-8")

    assert 'export ROBOLINEAGE_TASKS_ROOT="${ROBOLINEAGE_TASKS_ROOT:-$SCRIPT_DIR/tasks}"' in run_sh
    assert 'TASK_DIR="${ROBOLINEAGE_TASK_DIR:-$SCRIPT_DIR/.runtime/console}"' in run_sh
    assert 'TASK_DIR="$SCRIPT_DIR/tasks/${_TASK_ID}_${_TASK_TS}"' not in run_sh
    assert 'print(yaml.safe_load(f)[\'rollout\'][\'task_id\'])' not in run_sh
