from __future__ import annotations

import json
import os

import pytest

from robolineage_train.gate import ConfirmMergeGate, GateDecision, GateDeniedError

from .helpers import write_report


def test_issue_and_check_pass(tmp_path):
    report = write_report(tmp_path / "review_report.json")
    gate = ConfirmMergeGate(tmp_path / "tickets")
    ticket = gate.issue(
        issued_by="operator1",
        review_artifact_paths=[report],
        dataset_lock_version="v1",
        target_policy_version="1.2.0",
    )
    assert gate.check(
        ticket.ticket_id,
        expected_dataset="v1",
        expected_policy="1.2.0",
    ) == GateDecision.PASS


def test_check_rejects_dataset_mismatch(tmp_path):
    report = write_report(tmp_path / "review_report.json")
    gate = ConfirmMergeGate(tmp_path / "tickets")
    ticket = gate.issue(
        issued_by="operator1",
        review_artifact_paths=[report],
        dataset_lock_version="v1",
        target_policy_version="1.2.0",
    )
    assert gate.check(
        ticket.ticket_id,
        expected_dataset="v2",
        expected_policy="1.2.0",
    ) == GateDecision.FAIL


def test_check_rejects_tampered_ticket(tmp_path):
    report = write_report(tmp_path / "review_report.json")
    tickets_dir = tmp_path / "tickets"
    gate = ConfirmMergeGate(tickets_dir)
    ticket = gate.issue(
        issued_by="operator1",
        review_artifact_paths=[report],
        dataset_lock_version="v1",
        target_policy_version="1.2.0",
    )
    path = tickets_dir / f"{ticket.ticket_id}.json"
    os.chmod(path, 0o644)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_policy_version"] = "9.9.9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert gate.check(
        ticket.ticket_id,
        expected_dataset="v1",
        expected_policy="1.2.0",
    ) == GateDecision.FAIL


def test_check_rejects_malformed_ticket_file(tmp_path):
    tickets_dir = tmp_path / "tickets"
    tickets_dir.mkdir()
    (tickets_dir / "bad.json").write_text("{not json", encoding="utf-8")
    gate = ConfirmMergeGate(tickets_dir)
    assert gate.check(
        "bad",
        expected_dataset="v1",
        expected_policy="1.2.0",
    ) == GateDecision.FAIL


def test_issue_without_reports_is_not_a_pass(tmp_path):
    gate = ConfirmMergeGate(tmp_path / "tickets")
    ticket = gate.issue(
        issued_by="operator1",
        review_artifact_paths=[],
        dataset_lock_version="v1",
        target_policy_version="1.2.0",
    )
    assert gate.check(
        ticket.ticket_id,
        expected_dataset="v1",
        expected_policy="1.2.0",
    ) == GateDecision.FAIL


def test_require_pass_raises_on_missing_ticket(tmp_path):
    gate = ConfirmMergeGate(tmp_path / "tickets")
    with pytest.raises(GateDeniedError):
        gate.require_pass(
            "missing",
            expected_dataset="v1",
            expected_policy="1.2.0",
        )
