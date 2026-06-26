from __future__ import annotations

import sys

import pytest

from robolineage_train.trainer import Trainer, TrainingConfig

from .helpers import write_dataset_lock


def test_trainer_runs_command_and_finds_artifacts(tmp_path):
    lock_path = tmp_path / "dataset.lock"
    write_dataset_lock(lock_path)
    out_dir = tmp_path / "checkpoints" / "1.2.0"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            "out=Path('{output_dir}'); "
            "(out/'policy.bin').write_bytes(b'policy'); "
            "(out/'training_log.txt').write_text('ok'); "
            "print('training_steps=42')"
        ),
    ]
    result = Trainer(
        TrainingConfig(
            policy_version="1.2.0",
            architecture="diffusion_policy",
            dataset_lock_path=lock_path,
            output_dir=out_dir,
            command=command,
        )
    ).run()
    assert result.policy_bin_path == out_dir / "policy.bin"
    assert result.log_path == out_dir / "training_log.txt"
    assert result.steps == 42


def test_trainer_requires_policy_bin(tmp_path):
    lock_path = tmp_path / "dataset.lock"
    write_dataset_lock(lock_path)
    out_dir = tmp_path / "checkpoints" / "1.2.0"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; out=Path('{output_dir}'); (out/'training_log.txt').write_text('ok')",
    ]
    trainer = Trainer(
        TrainingConfig(
            policy_version="1.2.0",
            architecture="act_v2",
            dataset_lock_path=lock_path,
            output_dir=out_dir,
            command=command,
        )
    )
    with pytest.raises(FileNotFoundError, match="policy.bin"):
        trainer.run()


def test_trainer_requires_training_log(tmp_path):
    lock_path = tmp_path / "dataset.lock"
    write_dataset_lock(lock_path)
    out_dir = tmp_path / "checkpoints" / "1.2.0"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; out=Path('{output_dir}'); (out/'policy.bin').write_bytes(b'policy')",
    ]
    trainer = Trainer(
        TrainingConfig(
            policy_version="1.2.0",
            architecture="act_v2",
            dataset_lock_path=lock_path,
            output_dir=out_dir,
            command=command,
            training_steps=7,
        )
    )
    with pytest.raises(FileNotFoundError, match="training_log.txt"):
        trainer.run()
