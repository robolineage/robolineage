from __future__ import annotations

from pathlib import Path

import yaml


def test_training_framework_examples_have_required_commands() -> None:
    root = Path("configs/training_frameworks")
    profiles = sorted(root.glob("*.example.yaml"))

    assert {path.name for path in profiles} == {
        "act_hdf5.example.yaml",
        "diffusion_policy.example.yaml",
        "lerobot_vla.example.yaml",
    }
    for path in profiles:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["repo_root"].startswith("/path/to/")
        assert payload["dataset_command"]["args"]
        assert payload["train_command"]["args"]
        assert payload["eval_command"]["args"]
        assert "{dataset_output}" in " ".join(payload["dataset_command"]["args"] + payload["train_command"]["args"])
        assert "{checkpoint_path}" in " ".join(payload["eval_command"]["args"])
        assert payload["outputs"]["checkpoint_glob"].startswith("{checkpoint_dir}")
