from __future__ import annotations
from typing import Optional
import numpy as np

from .types import CameraParams, TrajectoryPoint


class PinholeProjector:
    """Projects 3D end-effector positions (robot base frame) to 2D image pixels.

    Standard pinhole projection pipeline:
      1. Apply extrinsic transform: p_cam = T_cam_from_robot @ [x, y, z, 1]ᵀ
      2. Guard z_cam > 0 — point must be in front of camera
      3. Apply intrinsics:
             u = fx * (x_cam / z_cam) + cx
             v = fy * (y_cam / z_cam) + cy
      4. Clip to image bounds — return None if outside

    Usage:
        proj = PinholeProjector(camera_params, image_width=640, image_height=480)
        pixel = proj.project(TrajectoryPoint(x=0.3, y=0.0, z=0.7))
        # pixel is (u, v) or None
    """

    def __init__(
        self,
        camera: CameraParams,
        image_width: int = 640,
        image_height: int = 480,
    ) -> None:
        self._camera = camera
        self._w = image_width
        self._h = image_height

    def project(self, point: TrajectoryPoint) -> Optional[tuple[int, int]]:
        """Return (u, v) pixel coordinate, or None if behind camera / out of bounds."""
        p_robot = np.array([point.x, point.y, point.z, 1.0], dtype=np.float64)
        p_cam = self._camera.T_cam_from_robot @ p_robot
        x_c, y_c, z_c = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])

        if z_c <= 0.0:
            return None  # behind or on the image plane

        u = int(self._camera.fx * x_c / z_c + self._camera.cx)
        v = int(self._camera.fy * y_c / z_c + self._camera.cy)

        if u < 0 or u >= self._w or v < 0 or v >= self._h:
            return None  # outside image boundary

        return (u, v)

    def project_trajectory(
        self, points: list[TrajectoryPoint]
    ) -> list[Optional[tuple[int, int]]]:
        """Project every point in the list. None entries = invisible / out of bounds."""
        return [self.project(p) for p in points]
