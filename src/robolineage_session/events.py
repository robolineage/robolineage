"""Append-only event logging and in-memory broadcasting."""
from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from robolineage_contracts.session import EventEnvelope, EventSource


def envelope_to_dict(env: EventEnvelope) -> dict:
    data = asdict(env)
    if isinstance(env.source, EventSource):
        data["source"] = env.source.value
    return data


class EventLogger:
    """Thread-safe JSONL writer for EventEnvelope records."""

    def __init__(self, target_path: Path) -> None:
        self.target_path = target_path
        self._lock = threading.RLock()
        self._closed = False
        self.target_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, env: EventEnvelope) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("event logger is closed")
            with self.target_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(envelope_to_dict(env), ensure_ascii=False, sort_keys=True))
                f.write("\n")

    def close(self) -> None:
        with self._lock:
            self._closed = True


class EventBroadcaster:
    """Small in-memory pub/sub queue set for SSE consumers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: set[queue.Queue[EventEnvelope | None]] = set()

    def broadcast(self, env: EventEnvelope) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(env)

    def subscribe(self) -> Iterator[EventEnvelope]:
        q: queue.Queue[EventEnvelope | None] = queue.Queue()
        with self._lock:
            self._subscribers.add(q)
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield item
        finally:
            with self._lock:
                self._subscribers.discard(q)

    def close(self) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for subscriber in subscribers:
            subscriber.put(None)
