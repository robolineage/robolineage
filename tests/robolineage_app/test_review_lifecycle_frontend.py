from __future__ import annotations

from pathlib import Path


def test_review_lifecycle_distinguishes_trainable_review_from_failure_pool():
    text = Path("frontend/src/views/ReviewLifecycleView.vue").read_text(encoding="utf-8")

    assert "review + trainable" in text
    assert "failure pool" in text
    assert "failure_pool_candidate" in text
