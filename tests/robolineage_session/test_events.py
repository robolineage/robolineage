import json
import threading
from pathlib import Path

import pytest

from robolineage_contracts.session import EventEnvelope, EventSource
from robolineage_session.events import EventLogger


def _env(i: int) -> EventEnvelope:
    return EventEnvelope(
        event="TEST",
        event_id=str(i),
        timestamp="2026-04-25T00:00:00.000Z",
        rollout_id="rollout",
        source=EventSource.AR,
        payload={"i": i},
    )


def test_event_logger_appends_jsonl(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path)

    logger.append(_env(1))
    logger.append(_env(2))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event_id"] for row in rows] == ["1", "2"]
    assert rows[0]["source"] == "ar"


def test_event_logger_thread_safe_count(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path)

    threads = [
        threading.Thread(target=logger.append, args=(_env(i),))
        for i in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(path.read_text(encoding="utf-8").splitlines()) == 20


def test_event_logger_rejects_after_close(tmp_path: Path):
    logger = EventLogger(tmp_path / "events.jsonl")

    logger.close()

    with pytest.raises(RuntimeError, match="closed"):
        logger.append(_env(1))
