from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .deploy import deploy_policy
from .discovery import CommandIntake, FrameworkDiscoveryAgent
from .framework_adapter import FrameworkAdapter, load_framework_profile
from .gate import ConfirmMergeGate
from .lifecycle import TrainingLifecycleRunner
from .policy_meta import PolicyMetaWriter
from .trainer import Trainer, TrainingConfig, _read_dataset_lock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RoboLineage T_update training and deployment gates."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="Run an external trainer and write policy.meta.json.",
    )
    train.add_argument("--dataset-lock", required=True, type=Path)
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)

    gate = subparsers.add_parser("gate", help="Manage CONFIRM_MERGE gate tickets.")
    gate_subparsers = gate.add_subparsers(dest="gate_command", required=True)
    issue = gate_subparsers.add_parser(
        "issue",
        help="Issue a manual CONFIRM_MERGE ticket.",
    )
    issue.add_argument("--operator", required=True)
    issue.add_argument("--dataset-lock", required=True, type=Path)
    issue.add_argument("--policy-version", required=True)
    issue.add_argument("--reports", required=True, nargs="+", type=Path)
    issue.add_argument("--tickets-dir", type=Path, default=Path("tickets"))

    deploy = subparsers.add_parser(
        "deploy",
        help="Write deployment.json after gate approval.",
    )
    deploy.add_argument("--policy-dir", required=True, type=Path)
    deploy.add_argument("--ticket", required=True)
    deploy.add_argument("--tickets-dir", type=Path, default=None)

    framework = subparsers.add_parser(
        "framework-run",
        help="Lightweight adapter for an existing dataset/train/eval repository.",
    )
    framework.add_argument("--profile", required=True, type=Path)
    framework.add_argument("--rollouts-root", required=True, type=Path)
    framework.add_argument("--workspace", required=True, type=Path)
    framework.add_argument("--dataset-version", required=True)
    framework.add_argument("--policy-version", required=True)
    framework.add_argument(
        "--include-decision",
        action="append",
        default=None,
        help="Dataset admission decision to include; repeatable. Default: accepted.",
    )
    framework.add_argument(
        "--include-rollout-id",
        action="append",
        default=None,
        help="Restrict framework run to selected rollout ids; repeatable.",
    )
    framework.set_defaults(symlink_selected=None)
    framework.add_argument(
        "--symlink-selected",
        dest="symlink_selected",
        action="store_true",
        help="Force selected_rollouts/ symlink staging.",
    )
    framework.add_argument(
        "--no-symlink-selected",
        dest="symlink_selected",
        action="store_false",
        help="Only write selected_rollouts.json; do not create selected_rollouts/ symlinks.",
    )

    discover = subparsers.add_parser(
        "discover-framework",
        help="Generate a RoboLineage framework profile from user-provided dataset/train/eval commands.",
    )
    discover.add_argument("--repo-root", required=True, type=Path)
    discover.add_argument("--output-dir", required=True, type=Path)
    discover.add_argument("--name", default=None)
    discover.add_argument("--framework-type", default=None)
    discover.add_argument("--dataset-command", default=None)
    discover.add_argument("--train-command", default=None)
    discover.add_argument("--eval-command", default=None)
    discover.add_argument("--fixed-input-dir", default=None)
    discover.add_argument("--checkpoint-glob", default=None)
    discover.add_argument("--train-log", default=None)
    discover.add_argument("--eval-result", default=None)
    discover.add_argument(
        "--llm-understanding",
        action="store_true",
        help="Use optional LLM repo understanding to refine framework type, outputs and log patterns.",
    )

    lifecycle = subparsers.add_parser(
        "lifecycle-run",
        help="Run post-review selected rollouts through dataset version, framework train/eval and policy metadata.",
    )
    lifecycle.add_argument("--profile", required=True, type=Path)
    lifecycle.add_argument("--rollouts-root", required=True, type=Path)
    lifecycle.add_argument("--datasets-root", required=True, type=Path)
    lifecycle.add_argument("--workspace-root", required=True, type=Path)
    lifecycle.add_argument("--policy-version", required=True)
    lifecycle.add_argument("--architecture", default=None)
    lifecycle.add_argument("--prev-lock", type=Path, default=None)
    lifecycle.add_argument("--include-decision", action="append", default=None)
    lifecycle.add_argument("--include-rollout-id", action="append", default=None)
    lifecycle.add_argument("--deploy-success-threshold", type=float, default=0.7)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = _load_training_config(args.config, args.dataset_lock, args.output_dir)
        result = Trainer(config).run()
        lock = _read_dataset_lock(args.dataset_lock)
        meta_path = PolicyMetaWriter().write(
            trainer_result=result,
            dataset_lock=lock,
            training_config=config,
        )
        print(json.dumps({"policy_meta": str(meta_path)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "gate" and args.gate_command == "issue":
        lock = _read_dataset_lock(args.dataset_lock)
        ticket = ConfirmMergeGate(args.tickets_dir).issue(
            issued_by=args.operator,
            review_artifact_paths=args.reports,
            dataset_lock_version=lock.version_id,
            target_policy_version=args.policy_version,
        )
        print(json.dumps(asdict(ticket), ensure_ascii=False, indent=2))
        return 0

    if args.command == "deploy":
        tickets_dir = args.tickets_dir or _default_tickets_dir(args.policy_dir)
        deployment_path = deploy_policy(
            policy_dir=args.policy_dir,
            ticket_id=args.ticket,
            tickets_dir=tickets_dir,
        )
        print(json.dumps({"deployment": str(deployment_path)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "framework-run":
        profile = load_framework_profile(args.profile)
        result = FrameworkAdapter(profile).run(
            rollouts_root=args.rollouts_root,
            workspace_dir=args.workspace,
            dataset_version=args.dataset_version,
            policy_version=args.policy_version,
            include_decisions=tuple(args.include_decision or ["accepted"]),
            include_rollout_ids=tuple(args.include_rollout_id) if args.include_rollout_id else None,
            symlink_selected=args.symlink_selected,
        )
        payload = json.loads(result.training_result_path.read_text(encoding="utf-8"))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "discover-framework":
        result = FrameworkDiscoveryAgent().discover(
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            name=args.name,
            framework_type=args.framework_type,
            commands=CommandIntake(
                dataset_command=args.dataset_command,
                train_command=args.train_command,
                eval_command=args.eval_command,
            ),
            fixed_input_dir=args.fixed_input_dir,
            checkpoint_glob=args.checkpoint_glob,
            train_log=args.train_log,
            eval_result=args.eval_result,
            enable_llm_understanding=args.llm_understanding,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "lifecycle-run":
        profile = load_framework_profile(args.profile)
        result = TrainingLifecycleRunner(
            profile=profile,
            rollouts_root=args.rollouts_root,
            datasets_root=args.datasets_root,
            workspace_root=args.workspace_root,
            prev_lock_path=args.prev_lock,
            include_decisions=tuple(args.include_decision or ["accepted"]),
            include_rollout_ids=tuple(args.include_rollout_id) if args.include_rollout_id else None,
            deploy_success_threshold=args.deploy_success_threshold,
        ).run(
            policy_version=args.policy_version,
            architecture=args.architecture,
        )
        print(json.dumps({key: str(value) for key, value in asdict(result).items()}, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _load_training_config(
    config_path: Path,
    dataset_lock_path: Path,
    output_root: Path,
) -> TrainingConfig:
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    policy_version = str(raw["policy_version"])
    output_dir = Path(output_root) / policy_version
    command = raw.get("command")
    if not isinstance(command, list) or not all(
        isinstance(part, str) for part in command
    ):
        raise ValueError("training config command must be a list of strings")
    return TrainingConfig(
        policy_version=policy_version,
        architecture=str(raw["architecture"]),
        dataset_lock_path=Path(dataset_lock_path),
        output_dir=output_dir,
        command=command,
        training_steps=raw.get("training_steps"),
    )


def _default_tickets_dir(policy_dir: Path) -> Path:
    policy_dir = Path(policy_dir)
    if policy_dir.parent.name == "checkpoints":
        return policy_dir.parent.parent / "tickets"
    return policy_dir.parent / "tickets"


if __name__ == "__main__":
    raise SystemExit(main())
