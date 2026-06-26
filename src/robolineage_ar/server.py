from __future__ import annotations
from contextlib import asynccontextmanager
import logging
import time
import threading
from typing import Callable, Optional

import cv2
from fastapi import FastAPI, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .projector import PinholeProjector
from .renderer import FrameRenderer
from .types import CameraParams, RenderConfig, TrajectoryPoint
from .video_source import VideoSource

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class _PointIn(BaseModel):
    x: float
    y: float
    z: float


class _TrajectoryIn(BaseModel):
    points: list[_PointIn]


# ---------------------------------------------------------------------------
# Thread-safe trajectory state
# ---------------------------------------------------------------------------

class _TrajectoryState:
    """Holds the current predicted trajectory. Thread-safe for concurrent
    POST /trajectory and GET /stream access."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: list[TrajectoryPoint] = []

    def set(self, points: list[TrajectoryPoint]) -> None:
        with self._lock:
            self._points = points

    def get(self) -> list[TrajectoryPoint]:
        with self._lock:
            return list(self._points)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    video_source: VideoSource,
    camera: CameraParams,
    render_config: RenderConfig,
    image_width: int = 640,
    image_height: int = 480,
    trajectory_gate: Optional[Callable[[], bool]] = None,
) -> FastAPI:
    """Create and return the FastAPI application.

    Args:
        video_source:  Any VideoSource — FileVideoSource, LiveCameraSource, or
                       SyntheticVideoSource for testing.
        camera:        Calibrated camera parameters (intrinsics + extrinsics).
        render_config: Visual style for the AR overlay.
        image_width:   Expected frame width in pixels (default 640).
        image_height:  Expected frame height in pixels (default 480).
        trajectory_gate: Optional gate. When non-None, ``POST /trajectory``
                       returns 204 (silent drop) unless the gate returns True.
                       Session runtime uses this to enforce "only accept trajectories
                       while a B1/B2 rollout is COLLECTING".

    Endpoints:
        POST /trajectory   — update the current predicted trajectory
        GET  /stream       — MJPEG video stream with AR overlay
        GET  /mjpeg        — alias for /stream, used by the session service
        GET  /health       — liveness check + current trajectory size

    Direct ROS2 data paths are handled outside the AR app. ``POST /trajectory``
    remains the stable overlay update API for policy or visualization
    producers that want to draw future trajectory points.
    """
    def shutdown_overlay_consumers() -> None:
        """Compatibility no-op for session shutdown."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            shutdown_overlay_consumers()

    app = FastAPI(
        title="RoboLineage AR Trajectory Renderer",
        version="0.1.0",
        lifespan=lifespan,
    )
    state = _TrajectoryState()
    projector = PinholeProjector(camera, image_width=image_width, image_height=image_height)
    renderer = FrameRenderer(render_config)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "trajectory_points": len(state.get()),
        }

    @app.post("/trajectory")
    def update_trajectory(body: _TrajectoryIn):
        """Replace the current trajectory with a new one.

        The policy inference process should POST here whenever it
        produces a new action sequence.  Older trajectories are
        atomically replaced — no partial updates.
        """
        if trajectory_gate is not None and not trajectory_gate():
            return Response(status_code=204)
        points = [
            TrajectoryPoint(x=p.x, y=p.y, z=p.z)
            for p in body.points
        ]
        # Enforce max_points limit (keep the most recent ones)
        trimmed = points[-render_config.max_points:]
        state.set(trimmed)
        return {"accepted": len(trimmed)}

    def _stream_response(max_frames: Optional[int]) -> StreamingResponse:
        def frame_generator():
            count = 0
            while max_frames is None or count < max_frames:
                frame = video_source.read()
                if frame is None:
                    if video_source.is_live():
                        time.sleep(0.01)
                        continue
                    break

                pixels = projector.project_trajectory(state.get())
                rendered = renderer.render(frame, pixels)

                ok, encoded = cv2.imencode(
                    ".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not ok:
                    continue

                data = encoded.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(data)}\r\n\r\n".encode()
                    + data
                    + b"\r\n"
                )
                count += 1

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream")
    def stream(max_frames: Optional[int] = Query(default=None)):
        """MJPEG stream.

        Open in a browser: http://<host>:8765/stream
        Use in an <img> tag: <img src="http://<host>:8765/stream">

        Query param:
            max_frames (int, optional): stop after N frames (for testing).
                       Omit for an infinite stream.
        """
        return _stream_response(max_frames)

    @app.get("/mjpeg")
    def mjpeg(max_frames: Optional[int] = Query(default=None)):
        """Session-service alias for the MJPEG stream."""
        return _stream_response(max_frames)

    app.state.shutdown_overlay_consumers = shutdown_overlay_consumers

    return app
