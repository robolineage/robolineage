from __future__ import annotations

import json

import pytest

from robolineage_train.deploy import deploy_policy
from robolineage_train.gate import ConfirmMergeGate, GateDeniedError
from robolineage_train.policy_meta import PolicyMetaWriter

from .helpers import make_dataset_lock, write_report
from .test_policy_meta import _config, _result


def test_deploy_requires_matching_gate_and_writes_deployment_record(tmp_path):
    lock = make_dataset_lock("v1")
    config = _config(tmp_path)
    meta_path = PolicyMetaWriter().write(
        trainer_result=_result(tmp_path),
        dataset_lock=lock,
        training_config=config,
    )
    original_meta = meta_path.read_text(encoding="utf-8")
    report = write_report(tmp_path / "review_report.json")
    ticket = ConfirmMergeGate(tmp_path / "tickets").issue(
        issued_by="operator1",
        review_artifact_paths=[report],
        dataset_lock_version="v1",
        target_policy_version="1.2.0",
    )

    deployment = deploy_policy(
        policy_dir=config.output_dir,
        ticket_id=ticket.ticket_id,
        tickets_dir=tmp_path / "tickets",
    )

    payload = json.loads(deployment.read_text(encoding="utf-8"))
    assert payload["policy_version"] == "1.2.0"
    assert payload["dataset_version"] == "v1"
    assert payload["ticket_id"] == ticket.ticket_id
    assert meta_path.read_text(encoding="utf-8") == original_meta
    assert oct(deployment.stat().st_mode & 0o777) == "0o444"


def test_deploy_rejects_unmatched_gate(tmp_path):
    lock = make_dataset_lock("v1")
    config = _config(tmp_path)
    PolicyMetaWriter().write(
        trainer_result=_result(tmp_path),
        dataset_lock=lock,
        training_config=config,
    )
    report = write_report(tmp_path / "review_report.json")
    ticket = ConfirmMergeGate(tmp_path / "tickets").issue(
        issued_by="operator1",
        review_artifact_paths=[report],
        dataset_lock_version="v2",
        target_policy_version="1.2.0",
    )
    with pytest.raises(GateDeniedError):
        deploy_policy(
            policy_dir=config.output_dir,
            ticket_id=ticket.ticket_id,
            tickets_dir=tmp_path / "tickets",
        )
