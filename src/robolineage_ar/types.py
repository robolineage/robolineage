from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class TrajectoryPoint:
    """One predicted end-effector position in robot base frame (metres).

    Coordinates follow the robot manufacturer's base-frame convention.
    x/y/z are in metres. color is BGR (OpenCV convention).
    """
    x: float
    y: float
    z: float
    color: Tuple[int, int, int] = (0, 0, 255)  # BGR — default red


@dataclass
class CameraParams:
    """Pinhole camera intrinsics + extrinsic transform from robot base to camera frame.

    T_cam_from_robot is a 4×4 homogeneous matrix.
    A point p_robot in robot base coordinates is projected via:
        p_cam = T_cam_from_robot @ [x, y, z, 1]ᵀ

    Default is identity — camera frame = robot base frame.
    In production, replace with real calibration values obtained from
    camera_calibration.yaml or the robot's calibration routine.

    Typical calibration for a 640×480 head-mounted RGB-D camera:
        fx ~= 600, fy ~= 600, cx ~= 320, cy ~= 240
    """
    fx: float
    fy: float
    cx: float
    cy: float
    T_cam_from_robot: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )


@dataclass
class RenderConfig:
    """Visual style for the trajectory overlay drawn on each video frame."""
    point_radius: int = 6          # pixels — drawn circle radius at each EE point
    line_thickness: int = 2        # pixels — connecting line width
    trajectory_color: Tuple[int, int, int] = (0, 0, 255)  # BGR — default red
    max_points: int = 50           # discard older points beyond this count
