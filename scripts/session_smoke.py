#!/usr/bin/env python3
"""Session smoke test: START -> mock snapshots -> STOP -> SUBMIT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from robolineage_session.api import create_app
from robolineage_session.session import SessionRegistry


def _snapshot(frame_id: int, risk_level: str = "low") -> dict:
    return {
        "timestamp": float(frame_id),
        "frame_id": frame_id,
        "progress": "advancing",
        "risk_level": risk_level,
        "phase": "approach",
        "imminent_failure": risk_level == "high",
        "confidence": 0.9,
        "needs_review": False,
        "raw_response": "smoke",
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts/session_smoke.py")
    parser.add_argument("--data-root", type=Path, default=Path("data/rollouts"))
    parser.add_argument("--runtime-root", type=Path, default=Path("data/runtime"))
    args = parser.parse_args()

    registry = SessionRegistry()
    client = TestClient(create_app(
        data_root=args.data_root,
        runtime_root=args.runtime_root,
        registry=registry,
    ))

    started = client.post("/events", json={
        "event": "START_COLLECTING",
        "payload": {
            "task_id": "smoke_task",
            "mode": "B1",
            "operator_id": "smoke_operator",
            "policy_version": "1.0.0",
        },
    })
    started.raise_for_status()
    session = registry.require_current()

    session.runtime_dir.mkdir(parents=True, exist_ok=True)
    with (session.runtime_dir / "snapshots.jsonl").open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps(_snapshot(i), ensure_ascii=False) + "\n")

    client.post("/events", json={"event": "STOP_COLLECTING"}).raise_for_status()
    submitted = client.post("/events", json={"event": "SUBMIT_ROLLOUT"})
    submitted.raise_for_status()

    rollout_dir = session.rollout_dir
    required = [
        rollout_dir / ".closed",
        rollout_dir / "events.jsonl",
        rollout_dir / "snapshots.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing smoke outputs: {missing}")

    event_count = len((rollout_dir / "events.jsonl").read_text(encoding="utf-8").splitlines())
    if event_count < 5:
        raise RuntimeError(f"expected at least 5 events, got {event_count}")

    print(f"Smoke OK: {rollout_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
