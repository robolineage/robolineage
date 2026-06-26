from __future__ import annotations

from fastapi.testclient import TestClient

from robolineage_session.api import create_app


def test_robot_onboarding_callback_is_exposed() -> None:
    app = create_app(
        on_robot_onboard=lambda body: {
            "status": "generated",
            "echo_note": body.get("robot_note"),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/robots/onboarding",
        json={"profile_yaml": "schema_version: RoboLineage.robot_profile.v1", "robot_note": "note"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "generated", "echo_note": "note"}
