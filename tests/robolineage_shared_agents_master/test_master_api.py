from __future__ import annotations

from fastapi.testclient import TestClient

from robolineage_session.api import create_app


def test_master_status_and_review_callbacks_are_exposed() -> None:
    app = create_app(
        on_master_status=lambda: {"state": {"current_stage": "task_understanding"}},
        on_master_review=lambda: {"state": {"current_stage": "deployment_governance"}},
    )
    client = TestClient(app)

    status = client.get("/master/status")
    assert status.status_code == 200
    assert status.json()["state"]["current_stage"] == "task_understanding"

    review = client.post("/master/review")
    assert review.status_code == 200
    assert review.json()["state"]["current_stage"] == "deployment_governance"
