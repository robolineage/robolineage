"""
FastAPI server unit tests — use SyntheticVideoSource, no camera required.
"""
import pytest
from fastapi.testclient import TestClient

from robolineage_ar.server import create_app
from robolineage_ar.types import CameraParams, RenderConfig
from robolineage_ar.video_source import SyntheticVideoSource


def make_client() -> TestClient:
    cam = CameraParams(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    cfg = RenderConfig()
    src = SyntheticVideoSource(height=480, width=640)
    app = create_app(video_source=src, camera=cam, render_config=cfg)
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_status_ok():
    c = make_client()
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_trajectory_points_initially_zero():
    c = make_client()
    assert c.get("/health").json()["trajectory_points"] == 0


# ---------------------------------------------------------------------------
# POST /trajectory
# ---------------------------------------------------------------------------

def test_post_trajectory_returns_200_and_accepted_count():
    c = make_client()
    resp = c.post("/trajectory", json={"points": [
        {"x": 0.0, "y": 0.0, "z": 1.0},
        {"x": 0.1, "y": 0.0, "z": 1.0},
    ]})
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 2


def test_post_trajectory_updates_health_count():
    c = make_client()
    c.post("/trajectory", json={"points": [{"x": 0.0, "y": 0.0, "z": 1.0}]})
    assert c.get("/health").json()["trajectory_points"] == 1


def test_post_trajectory_replaces_previous():
    c = make_client()
    c.post("/trajectory", json={"points": [{"x": 0.0, "y": 0.0, "z": 1.0}] * 5})
    c.post("/trajectory", json={"points": [{"x": 0.0, "y": 0.0, "z": 1.0}] * 2})
    assert c.get("/health").json()["trajectory_points"] == 2


def test_post_trajectory_enforces_max_points():
    cfg = RenderConfig(max_points=3)
    src = SyntheticVideoSource()
    cam = CameraParams(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    app = create_app(video_source=src, camera=cam, render_config=cfg)
    c = TestClient(app)
    c.post("/trajectory", json={"points": [{"x": 0.0, "y": 0.0, "z": 1.0}] * 10})
    assert c.get("/health").json()["trajectory_points"] == 3


def test_post_trajectory_rejects_missing_z():
    c = make_client()
    resp = c.post("/trajectory", json={"points": [{"x": 0.0, "y": 0.0}]})
    assert resp.status_code == 422


def test_post_trajectory_gate_returns_204_without_update():
    cam = CameraParams(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    app = create_app(
        video_source=SyntheticVideoSource(),
        camera=cam,
        render_config=RenderConfig(),
        trajectory_gate=lambda: False,
    )
    c = TestClient(app)

    resp = c.post("/trajectory", json={"points": [{"x": 0.0, "y": 0.0, "z": 1.0}]})

    assert resp.status_code == 204
    assert c.get("/health").json()["trajectory_points"] == 0


# ---------------------------------------------------------------------------
# GET /stream
# ---------------------------------------------------------------------------

def test_stream_content_type_is_multipart():
    c = make_client()
    with c.stream("GET", "/stream?max_frames=1") as resp:
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.headers["content-type"]


def test_stream_contains_jpeg_soi_marker():
    c = make_client()
    data = b""
    with c.stream("GET", "/stream?max_frames=1") as resp:
        for chunk in resp.iter_bytes():
            data += chunk
    assert b"\xff\xd8\xff" in data  # JPEG Start Of Image


def test_stream_with_trajectory_still_works():
    c = make_client()
    c.post("/trajectory", json={"points": [
        {"x": 0.0, "y": 0.0, "z": 1.0},
        {"x": 0.1, "y": 0.0, "z": 1.0},
    ]})
    data = b""
    with c.stream("GET", "/stream?max_frames=2") as resp:
        for chunk in resp.iter_bytes():
            data += chunk
    assert b"\xff\xd8\xff" in data
