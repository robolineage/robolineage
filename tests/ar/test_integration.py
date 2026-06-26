"""
Integration tests: real task_98/camera_h.mp4 + synthetic trajectory → MJPEG stream.
All tests are skipped if sample data is absent (e.g. on CI without data volume).
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from robolineage_ar.server import create_app
from robolineage_ar.types import CameraParams, RenderConfig
from robolineage_ar.video_source import FileVideoSource

REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLE_VIDEO = (
    REPO_ROOT
    / "data" / "task_98"
    / "027b72ff-fbf2-4f6b-ba1b-9433bbd103e4"
    / "videos" / "camera_h.mp4"
)

has_sample = SAMPLE_VIDEO.exists()

# Reasonable intrinsics for a 640×480 head-mounted RealSense camera (~60° HFoV)
CAM = CameraParams(fx=600.0, fy=600.0, cx=320.0, cy=240.0)


def make_client():
    src = FileVideoSource(SAMPLE_VIDEO)
    app = create_app(video_source=src, camera=CAM, render_config=RenderConfig())
    return TestClient(app)


def arc_trajectory(n=10) -> dict:
    """Synthetic EE arc: sweeps x from 0.1→0.4, z from 0.5→0.8 (in front of camera)."""
    points = []
    for i in range(n):
        t = i / max(n - 1, 1)
        points.append({"x": 0.1 + t * 0.3, "y": 0.0, "z": 0.5 + t * 0.3})
    return {"points": points}


@pytest.mark.skipif(not has_sample, reason="Sample video not present")
def test_health_with_real_video():
    assert make_client().get("/health").json()["status"] == "ok"


@pytest.mark.skipif(not has_sample, reason="Sample video not present")
def test_full_pipeline_produces_jpeg_stream():
    c = make_client()
    c.post("/trajectory", json=arc_trajectory(10))
    assert c.get("/health").json()["trajectory_points"] == 10

    data = b""
    with c.stream("GET", "/stream?max_frames=3") as resp:
        assert resp.status_code == 200
        for chunk in resp.iter_bytes():
            data += chunk

    assert b"\xff\xd8\xff" in data  # JPEG SOI
    assert b"\xff\xd9" in data      # JPEG EOI


@pytest.mark.skipif(not has_sample, reason="Sample video not present")
def test_stream_works_without_trajectory():
    c = make_client()
    data = b""
    with c.stream("GET", "/stream?max_frames=2") as resp:
        for chunk in resp.iter_bytes():
            data += chunk
    assert b"\xff\xd8\xff" in data


@pytest.mark.skipif(not has_sample, reason="Sample video not present")
def test_trajectory_replacement_mid_stream():
    c = make_client()
    c.post("/trajectory", json=arc_trajectory(5))
    c.post("/trajectory", json=arc_trajectory(15))
    assert c.get("/health").json()["trajectory_points"] == 15
