from __future__ import annotations
from typing import Optional
import cv2
import numpy as np

from .types import RenderConfig


class FrameRenderer:
    """Draws a projected trajectory onto a BGR video frame using OpenCV.

    Input is the output of PinholeProjector.project_trajectory():
    a list of Optional[(u, v)] pixel coordinates.

    None entries represent invisible / out-of-frame points and break
    the connecting line — matching how a trajectory disappears when
    part of it goes behind the camera or out of view.

    Usage:
        renderer = FrameRenderer(RenderConfig())
        overlay = renderer.render(bgr_frame, projected_pixels)
        # overlay is a new frame — original is not modified
    """

    def __init__(self, config: RenderConfig) -> None:
        self._cfg = config

    def render(
        self,
        frame: np.ndarray,
        pixels: list[Optional[tuple[int, int]]],
    ) -> np.ndarray:
        """Return a new BGR frame with trajectory overlay. Input frame is not modified."""
        out = frame.copy()

        # --- Draw connecting lines first (so circles render on top) ---
        prev: Optional[tuple[int, int]] = None
        for px in pixels:
            if px is not None and prev is not None:
                cv2.line(out, prev, px, self._cfg.trajectory_color, self._cfg.line_thickness)
            prev = px  # None breaks the line chain (gap in trajectory)

        # --- Draw filled circles at each visible point ---
        for px in pixels:
            if px is not None:
                cv2.circle(out, px, self._cfg.point_radius, self._cfg.trajectory_color, -1)

        return out
