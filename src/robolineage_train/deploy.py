from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .gate import ConfirmMergeGate


@dataclass(frozen=True)
class DeploymentRecord:
    policy_version: str
    dataset_version: str
    ticket_id: str
    deployed_at: str
    gate_signature: str


def deploy_policy(
    *,
    policy_dir: Path,
    ticket_id: str,
    tickets_dir: Path,
) -> Path:
    policy_dir = Path(policy_dir)
    meta = json.loads((policy_dir / "policy.meta.json").read_text(encoding="utf-8"))
    policy_version = str(meta["version_id"])
    dataset_version = str(meta["trained_on_dataset"])

    gate = ConfirmMergeGate(tickets_dir)
    ticket = gate.require_pass(
        ticket_id,
        expected_dataset=dataset_version,
        expected_policy=policy_version,
    )
    target = policy_dir / "deployment.json"
    if target.exists():
        raise FileExistsError(f"deployment.json already exists: {target}")
    record = DeploymentRecord(
        policy_version=policy_version,
        dataset_version=dataset_version,
        ticket_id=ticket.ticket_id,
        deployed_at=datetime.now(timezone.utc).isoformat(),
        gate_signature=ticket.signature,
    )
    tmp = target.with_name("deployment.json.tmp")
    tmp.write_text(
        json.dumps(asdict(record), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    target.chmod(0o444)
    return target
