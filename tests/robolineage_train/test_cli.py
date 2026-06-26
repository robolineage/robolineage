from __future__ import annotations

import json
import sys

from robolineage_train.__main__ import main

from .helpers import write_dataset_lock, write_report


def test_train_gate_deploy_cli_flow(tmp_path):
    lock_path = tmp_path / "dataset.lock"
    write_dataset_lock(lock_path, "v1")
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "\n".join(
            [
                "policy_version: 1.2.0",
                "architecture: diffusion_policy",
                "training_steps: 11",
                "command:",
                f"  - {json.dumps(sys.executable)}",
                "  - -c",
                "  - \"from pathlib import Path; out=Path('{output_dir}'); "
                "(out/'policy.bin').write_bytes(b'policy'); "
                "(out/'training_log.txt').write_text('ok')\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    checkpoints = tmp_path / "checkpoints"
    assert main([
        "train",
        "--dataset-lock", str(lock_path),
        "--config", str(config_path),
        "--output-dir", str(checkpoints),
    ]) == 0
    policy_dir = checkpoints / "1.2.0"
    assert (policy_dir / "policy.meta.json").exists()

    report = write_report(tmp_path / "review_report.json")
    tickets_dir = tmp_path / "tickets"
    assert main([
        "gate", "issue",
        "--operator", "operator1",
        "--dataset-lock", str(lock_path),
        "--policy-version", "1.2.0",
        "--reports", str(report),
        "--tickets-dir", str(tickets_dir),
    ]) == 0
    ticket_id = next(tickets_dir.glob("*.json")).stem

    assert main([
        "deploy",
        "--policy-dir", str(policy_dir),
        "--ticket", ticket_id,
        "--tickets-dir", str(tickets_dir),
    ]) == 0
    assert (policy_dir / "deployment.json").exists()
