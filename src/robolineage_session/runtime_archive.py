"""Archive runtime files into the rollout directory."""
from __future__ import annotations

from pathlib import Path


def archive(runtime_dir: Path, rollout_dir: Path) -> Path:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    target = rollout_dir / "snapshots.jsonl"
    if target.exists():
        raise FileExistsError(target)

    source = runtime_dir / "snapshots.jsonl"
    if not source.exists() or source.stat().st_size == 0:
        target.write_text("", encoding="utf-8")
        return target

    source.replace(target)
    return target
