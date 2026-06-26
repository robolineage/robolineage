from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from robolineage_contracts.pipeline import DatasetLock, compute_manifest_sha256

from .lock_writer import DatasetLockWriter
from .version import next_version_id


class DatasetUpdater:
    def __init__(self, lock_writer: DatasetLockWriter | None = None) -> None:
        self.lock_writer = lock_writer or DatasetLockWriter()

    def update(
        self,
        *,
        train_manifest_path: Path,
        prev_lock_path: Path | None,
        out_dir: Path,
        changelog: str = "",
    ) -> DatasetLock:
        new_entries = _read_jsonl(train_manifest_path)
        prev_lock = _read_lock(prev_lock_path) if prev_lock_path is not None else None
        prev_entries = _read_jsonl(prev_lock_path.parent / "manifest.jsonl") if prev_lock_path else []

        merged_entries = _merge_manifest_entries(prev_entries, new_entries)
        new_version = next_version_id(prev_lock.version_id if prev_lock is not None else None)
        target_dir = Path(out_dir) / new_version
        lock_path = target_dir / "dataset.lock"
        if lock_path.exists():
            raise FileExistsError(f"dataset.lock already exists: {lock_path}")

        rollout_ids = tuple(entry["rollout_id"] for entry in merged_entries)
        lock = DatasetLock(
            version_id=new_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            base_version_id=prev_lock.version_id if prev_lock is not None else None,
            included_rollout_ids=rollout_ids,
            total_samples=len(merged_entries),
            manifest_sha256=compute_manifest_sha256(merged_entries),
            changelog=changelog,
        )

        target_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl_atomic(target_dir / "manifest.jsonl", merged_entries)
        _write_text_atomic(target_dir / "README.md", self._render_readme(lock))
        self.lock_writer.write(lock, target_dir)
        return lock

    @staticmethod
    def _render_readme(lock: DatasetLock) -> str:
        base = lock.base_version_id or "None"
        return "\n".join(
            [
                f"# Dataset {lock.version_id}",
                "",
                f"- created_at: `{lock.created_at}`",
                f"- base_version_id: `{base}`",
                f"- total_samples: `{lock.total_samples}`",
                f"- manifest_sha256: `{lock.manifest_sha256}`",
                f"- changelog: {lock.changelog or '(empty)'}",
                "",
            ]
        )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_lock(path: Path) -> DatasetLock:
    raw = json.loads(path.read_text())
    raw["included_rollout_ids"] = tuple(raw["included_rollout_ids"])
    return DatasetLock(**raw)


def _merge_manifest_entries(prev_entries: list[dict], new_entries: list[dict]) -> list[dict]:
    by_rollout: dict[str, dict] = {}
    for entry in prev_entries + new_entries:
        rollout_id = entry["rollout_id"]
        by_rollout[rollout_id] = entry
    return [by_rollout[key] for key in sorted(by_rollout)]


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _write_text_atomic(path, text)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
