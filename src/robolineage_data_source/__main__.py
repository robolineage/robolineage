"""End-to-end entrypoint.

Two modes:
  `python -m robolineage_data_source`                     → MockAdapter health smoke test
  `python -m robolineage_data_source --config path.yaml`  → Orchestrator run
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from robolineage_data_source.adapters.mock import MockAdapter
from robolineage_data_source.config.loader import load_config
from robolineage_data_source.orchestrator import Orchestrator


def _run_smoke() -> int:
    adapter = MockAdapter(topic="mock/test", rate_hz=10.0)
    adapter.start()
    try:
        for i in range(10):
            time.sleep(0.1)
            health = adapter.health()
            print(f"[{i}] state={health.state.value} meta={dict(health.meta)}")
    finally:
        adapter.stop()
    health = adapter.health()
    print(f"adapter health: state={health.state.value} "
          f"last_sample_mono_ns={health.last_sample_mono_ns}")
    return 0


def _run_config(config_path: str, mode: str) -> int:
    cfg = load_config(config_path)
    orch = Orchestrator(cfg, recorder_mode=mode)
    print(f"starting rollout {orch.rollout_id}")

    stopping = {"flag": False}
    def handler(signum, frame):
        stopping["flag"] = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    orch.start()
    try:
        while not stopping["flag"]:
            time.sleep(0.5)
    finally:
        orch.stop()
    print(f"rollout {orch.rollout_id} stopped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="robolineage_data_source")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=("none", "rosbag"),
        default="rosbag",
        help="recorder output mode when --config is provided",
    )
    args = parser.parse_args()
    if args.config:
        return _run_config(args.config, args.mode)
    return _run_smoke()


if __name__ == "__main__":
    sys.exit(main())
