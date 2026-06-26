from __future__ import annotations

import json

from robolineage_dataset.__main__ import main


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_update_and_diff_cli(tmp_path, capsys):
    manifest = tmp_path / "train_manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "export_id": "r1",
                "rollout_id": "r1",
                "sample_dir": "rollouts/r1",
                "review_score": "A",
                "confidence": 0.9,
                "l1_phases": None,
                "reasons": ["accepted"],
            }
        ],
    )

    assert main(["update", "--train-manifest", str(manifest), "--out", str(tmp_path / "datasets")]) == 0
    out = capsys.readouterr().out
    assert '"version_id": "v1"' in out

    lock = tmp_path / "datasets" / "v1" / "dataset.lock"
    assert main(["diff", str(lock), str(lock)]) == 0
    out = capsys.readouterr().out
    assert '"total_delta": 0' in out
