from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from robolineage_contracts.pipeline import DatasetLock


class DatasetLockWriter:
    def write(self, lock: DatasetLock, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "dataset.lock"
        if target.exists():
            raise FileExistsError(f"dataset.lock already exists: {target}")

        payload = json.dumps(asdict(lock), ensure_ascii=False, indent=2) + "\n"
        tmp = target.with_name("dataset.lock.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(target)
        target.chmod(0o444)
        return target
