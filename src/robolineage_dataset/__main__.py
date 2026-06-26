from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from robolineage_contracts.pipeline import DatasetLock

from .diff import diff_locks
from .updater import DatasetUpdater


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage RoboLineage dataset versions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    update = subparsers.add_parser("update", help="Create the next dataset version.")
    update.add_argument("--train-manifest", required=True, type=Path)
    update.add_argument("--prev-lock", type=Path, default=None)
    update.add_argument("--out", required=True, type=Path)
    update.add_argument("--changelog", default="")

    diff = subparsers.add_parser("diff", help="Diff two dataset.lock files.")
    diff.add_argument("from_lock", type=Path)
    diff.add_argument("to_lock", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "update":
        lock = DatasetUpdater().update(
            train_manifest_path=args.train_manifest,
            prev_lock_path=args.prev_lock,
            out_dir=args.out,
            changelog=args.changelog,
        )
        print(json.dumps(asdict(lock), ensure_ascii=False, indent=2))
        return 0

    if args.command == "diff":
        diff = diff_locks(_read_lock(args.from_lock), _read_lock(args.to_lock))
        print(json.dumps(asdict(diff), ensure_ascii=False, indent=2))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _read_lock(path: Path) -> DatasetLock:
    raw = json.loads(path.read_text())
    raw["included_rollout_ids"] = tuple(raw["included_rollout_ids"])
    return DatasetLock(**raw)


if __name__ == "__main__":
    raise SystemExit(main())
