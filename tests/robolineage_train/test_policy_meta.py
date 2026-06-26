from __future__ import annotations

import json
import os

import pytest

from robolineage_train.policy_meta import PolicyMetaWriter
from robolineage_train.trainer import TrainingConfig, TrainingResult

from .helpers import make_dataset_lock


def _config(tmp_path, dataset_lock_path=None):
    return TrainingConfig(
        policy_version="1.2.0",
        architecture="diffusion_policy",
        dataset_lock_path=dataset_lock_path or tmp_path / "dataset.lock",
        output_dir=tmp_path / "checkpoints" / "1.2.0",
        command=["echo", "unused"],
    )


def _result(tmp_path):
    out = tmp_path / "checkpoints" / "1.2.0"
    out.mkdir(parents=True, exist_ok=True)
    policy = out / "policy.bin"
    log = out / "training_log.txt"
    policy.write_bytes(b"policy")
    log.write_text("ok", encoding="utf-8")
    return TrainingResult(policy_bin_path=policy, log_path=log, steps=42)


def test_write_policy_meta_is_readonly_and_bound_to_dataset(tmp_path):
    lock = make_dataset_lock("v7")
    path = PolicyMetaWriter().write(
        trainer_result=_result(tmp_path),
        dataset_lock=lock,
        training_config=_config(tmp_path),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version_id"] == "1.2.0"
    assert payload["trained_on_dataset"] == "v7"
    assert payload["deployed"] is False
    assert oct(path.stat().st_mode & 0o777) == "0o444"


def test_write_policy_meta_refuses_overwrite(tmp_path):
    writer = PolicyMetaWriter()
    writer.write(
        trainer_result=_result(tmp_path),
        dataset_lock=make_dataset_lock(),
        training_config=_config(tmp_path),
    )
    with pytest.raises(FileExistsError):
        writer.write(
            trainer_result=_result(tmp_path),
            dataset_lock=make_dataset_lock(),
            training_config=_config(tmp_path),
        )


def test_verify_integrity_reports_dataset_mismatch_after_tamper(tmp_path):
    lock = make_dataset_lock("v1")
    path = PolicyMetaWriter().write(
        trainer_result=_result(tmp_path),
        dataset_lock=lock,
        training_config=_config(tmp_path),
    )
    os.chmod(path, 0o644)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["trained_on_dataset"] = "v999"
    path.write_text(json.dumps(payload), encoding="utf-8")
    issues = PolicyMetaWriter.verify_integrity(path, lock)
    assert [issue.code for issue in issues] == ["policy_source_dataset_mismatch"]


def test_invalid_semver_is_rejected(tmp_path):
    bad_config = TrainingConfig(
        policy_version="v1",
        architecture="diffusion_policy",
        dataset_lock_path=tmp_path / "dataset.lock",
        output_dir=tmp_path / "checkpoints" / "v1",
        command=["echo", "unused"],
    )
    with pytest.raises(ValueError, match="SemVer"):
        PolicyMetaWriter().write(
            trainer_result=_result(tmp_path),
            dataset_lock=make_dataset_lock(),
            training_config=bad_config,
        )
