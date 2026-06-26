from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class GateDeniedError(RuntimeError):
    pass


class GateDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"


@dataclass(frozen=True)
class GateTicket:
    ticket_id: str
    issued_at: str
    issued_by: str
    review_artifact_hashes: tuple[str, ...]
    dataset_lock_version: str
    target_policy_version: str
    signature: str


class ConfirmMergeGate:
    def __init__(self, tickets_dir: Path) -> None:
        self.tickets_dir = Path(tickets_dir)

    def issue(
        self,
        *,
        issued_by: str,
        review_artifact_paths: list[Path],
        dataset_lock_version: str,
        target_policy_version: str,
    ) -> GateTicket:
        ticket = GateTicket(
            ticket_id=str(uuid.uuid4()),
            issued_at=datetime.now(timezone.utc).isoformat(),
            issued_by=issued_by,
            review_artifact_hashes=tuple(
                _sha256_file(path) for path in review_artifact_paths
            ),
            dataset_lock_version=dataset_lock_version,
            target_policy_version=target_policy_version,
            signature="",
        )
        ticket = replace(ticket, signature=_signature(ticket))
        self.tickets_dir.mkdir(parents=True, exist_ok=True)
        path = self._ticket_path(ticket.ticket_id)
        if path.exists():
            raise FileExistsError(f"gate ticket already exists: {path}")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(_ticket_to_json(ticket), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
        path.chmod(0o444)
        return ticket

    def check(
        self,
        ticket_id: str,
        *,
        expected_dataset: str,
        expected_policy: str,
    ) -> GateDecision:
        if not self._ticket_path(ticket_id).exists():
            return GateDecision.PENDING
        ticket = self.load(ticket_id)
        if ticket is None:
            return GateDecision.FAIL
        if ticket.signature != _signature(replace(ticket, signature="")):
            return GateDecision.FAIL
        if ticket.dataset_lock_version != expected_dataset:
            return GateDecision.FAIL
        if ticket.target_policy_version != expected_policy:
            return GateDecision.FAIL
        if not ticket.review_artifact_hashes:
            return GateDecision.FAIL
        return GateDecision.PASS

    def require_pass(
        self,
        ticket_id: str,
        *,
        expected_dataset: str,
        expected_policy: str,
    ) -> GateTicket:
        decision = self.check(
            ticket_id,
            expected_dataset=expected_dataset,
            expected_policy=expected_policy,
        )
        if decision != GateDecision.PASS:
            raise GateDeniedError(
                f"CONFIRM_MERGE denied: ticket={ticket_id} decision={decision.value}"
            )
        ticket = self.load(ticket_id)
        if ticket is None:
            raise GateDeniedError(f"CONFIRM_MERGE denied: ticket={ticket_id}")
        return ticket

    def load(self, ticket_id: str) -> GateTicket | None:
        path = self._ticket_path(ticket_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["review_artifact_hashes"] = tuple(
                raw.get("review_artifact_hashes", ())
            )
            return GateTicket(**raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _ticket_path(self, ticket_id: str) -> Path:
        return self.tickets_dir / f"{ticket_id}.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ticket_to_json(ticket: GateTicket) -> dict:
    data = asdict(ticket)
    data["review_artifact_hashes"] = list(ticket.review_artifact_hashes)
    return data


def _canonical_ticket_payload(ticket: GateTicket) -> bytes:
    data = _ticket_to_json(replace(ticket, signature=""))
    return json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _signature(ticket: GateTicket) -> str:
    digest = hashlib.sha256(_canonical_ticket_payload(ticket)).hexdigest()
    return f"manual:{ticket.issued_by}:{digest}"
