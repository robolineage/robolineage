"""Shared SnapshotAssessment JSONL writer for offline and realtime VSA paths."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import ClassVar, TextIO

from robolineage_contracts.agents import SnapshotAssessment


class SnapshotWriter:
    """Append SnapshotAssessment records to one explicit JSONL file path."""

    _open_paths: ClassVar[set[Path]] = set()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._resolved_path = self.path.resolve()
        self._file: TextIO | None = None
        self._closed = False

        if self._resolved_path in self._open_paths:
            raise FileExistsError(f"SnapshotWriter is already open for {self.path}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        try:
            self._lock_file()
        except Exception:
            self._file.close()
            self._file = None
            raise
        self._open_paths.add(self._resolved_path)

    def write(self, snapshot: SnapshotAssessment) -> None:
        if self._closed or self._file is None:
            raise ValueError("Cannot write to a closed SnapshotWriter.")

        line = json.dumps(asdict(snapshot), ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._closed:
            return

        try:
            if self._file is not None:
                self._unlock_file()
                self._file.close()
        finally:
            self._file = None
            self._closed = True
            self._open_paths.discard(self._resolved_path)

    def __enter__(self) -> "SnapshotWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _lock_file(self) -> None:
        if self._file is None:
            return
        try:
            import fcntl
        except ImportError:
            return

        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(f"SnapshotWriter is already locked for {self.path}") from exc

    def _unlock_file(self) -> None:
        if self._file is None:
            return
        try:
            import fcntl
        except ImportError:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
