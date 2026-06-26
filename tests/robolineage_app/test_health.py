"""Phase 5 T5 — health endpoint unit tests.

Verifies the systemd contract:
- All adapters OK / DEGRADED / NOT_STARTED → HTTP 200
- Any adapter FAILED → HTTP 503 (systemd Restart=on-failure trigger)
- meta is surfaced (jpeg_decode_failures etc.)
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from robolineage_data_source.sample import HealthState, HealthStatus


class _FakeRuntime:
    """Minimal runtime double — `create_health_app` only touches a few fields."""

    def __init__(self, adapter_states: dict):
        self.orchestrator = MagicMock()
        self.orchestrator._adapters = adapter_states
        self.session_app = None
        self._vsa_thread = None


def _adapter_with(state: HealthState, message: str = "", meta: dict | None = None):
    a = MagicMock()
    a.health.return_value = HealthStatus(state=state, message=message, meta=meta or {})
    return a


def test_health_returns_200_when_all_adapters_ok():
    from robolineage_app.health import create_health_app

    runtime = _FakeRuntime(
        {
            "ros2_arx_one": _adapter_with(HealthState.OK),
        }
    )
    client = TestClient(create_health_app(runtime))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["adapters"]["ros2_arx_one"]["state"] == "ok"


def test_health_returns_200_with_degraded_status_and_surfaces_meta():
    from robolineage_app.health import create_health_app

    runtime = _FakeRuntime(
        {
            "ros2_arx_one": _adapter_with(
                HealthState.DEGRADED,
                message="jpeg decode failed",
                meta={"jpeg_decode_failures": {"cam/camera_h/color": 5}},
            ),
        }
    )
    client = TestClient(create_health_app(runtime))
    resp = client.get("/health")
    # DEGRADED is still 200 (warn but don't restart); meta carries the detail
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["adapters"]["ros2_arx_one"]["meta"]["jpeg_decode_failures"][
        "cam/camera_h/color"
    ] == 5


def test_health_returns_503_when_any_adapter_failed():
    from robolineage_app.health import create_health_app

    runtime = _FakeRuntime(
        {
            "ros2_arx_one": _adapter_with(HealthState.OK),
            "broken": _adapter_with(HealthState.FAILED, message="rclpy died"),
        }
    )
    client = TestClient(create_health_app(runtime))
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "failed"


def test_health_handles_adapter_health_raising():
    """If adapter.health() itself throws, report failed + the exception text."""
    from robolineage_app.health import create_health_app

    bad = MagicMock()
    bad.health.side_effect = RuntimeError("synthetic boom")
    runtime = _FakeRuntime({"bad": bad})

    client = TestClient(create_health_app(runtime))
    resp = client.get("/health")
    assert resp.status_code == 503
    assert "synthetic boom" in resp.json()["adapters"]["bad"]["message"]


def test_health_no_orchestrator_returns_200_with_empty_adapter_map():
    """Edge case: services.data_source=false → orchestrator None."""
    from robolineage_app.health import create_health_app

    class _NoOrchRuntime:
        orchestrator = None
        session_app = None
        _vsa_thread = None

    client = TestClient(create_health_app(_NoOrchRuntime()))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["adapters"] == {}


def test_health_includes_ai_route_status_summary():
    from robolineage_app.health import create_health_app

    runtime = _FakeRuntime({})
    runtime.ai_routes_status = lambda: {
        "routes": {
            "TRAINING_MONITOR_LLM": {
                "configured": True,
                "implemented": True,
                "api_key_configured": True,
            }
        }
    }

    client = TestClient(create_health_app(runtime))
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["ai_routes"]["routes"]["TRAINING_MONITOR_LLM"]["configured"] is True
