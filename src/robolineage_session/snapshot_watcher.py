"""Polling tail reader for VSA snapshot JSONL output."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from robolineage_contracts.agents import SnapshotAssessment


class SnapshotWatcher:
    def __init__(self, path: Path, poll_interval: float = 1.0) -> None:
        self.path = path
        self.poll_interval = poll_interval
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def iter_new(self) -> Iterator[SnapshotAssessment]:
        offset = 0
        while not self._stopped:
            if not self.path.exists():
                time.sleep(self.poll_interval)
                continue

            with self.path.open("r", encoding="utf-8") as f:
                f.seek(offset)
                while not self._stopped:
                    line_start = f.tell()
                    line = f.readline()
                    if not line:
                        offset = f.tell()
                        break
                    if not line.endswith("\n"):
                        offset = line_start
                        break
                    offset = f.tell()
                    try:
                        payload = json.loads(line)
                        yield SnapshotAssessment(**payload)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
            time.sleep(self.poll_interval)
