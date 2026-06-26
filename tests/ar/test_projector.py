import numpy as np
from robolineage_ar.projector import PinholeProjector
from robolineage_ar.types import CameraParams, TrajectoryPoint

# 640×480, identity extrinsics
CAM = CameraParams(fx=600.0, fy=600.0, cx=320.0, cy=240.0)


def test_optical_axis_projects_to_principal_point():
    proj = PinholeProjector(CAM, image_width=640, image_height=480)
    result = proj.project(TrajectoryPoint(x=0.0, y=0.0, z=1.0))
    assert result == (320, 240)


def test_point_behind_camera_returns_none():
    proj = PinholeProjector(CAM, image_width=640, image_height=480)
    assert proj.project(TrajectoryPoint(x=0.0, y=0.0, z=-1.0)) is None


def test_point_at_z_zero_returns_none():
    proj = PinholeProjector(CAM, image_width=640, image_height=480)
    assert proj.project(TrajectoryPoint(x=0.0, y=0.0, z=0.0)) is None


def test_positive_x_offset_projects_right_of_centre():
    proj = PinholeProjector(CAM, image_width=640, image_height=480)
    result = proj.project(TrajectoryPoint(x=0.1, y=0.0, z=1.0))
    assert result is not None
    assert result[0] > 320


def test_positive_y_offset_projects_below_centre():
    proj = PinholeProjector(CAM, image_width=640, image_height=480)
    result = proj.project(TrajectoryPoint(x=0.0, y=0.1, z=1.0))
    assert result is not None
    assert result[1] > 240


def test_far_off_axis_point_returns_none():
    proj = PinholeProjector(CAM, image_width=640, image_height=480)
    assert proj.project(TrajectoryPoint(x=10.0, y=0.0, z=1.0)) is None


def test_project_trajectory_preserves_length():
    proj = PinholeProjector(CAM, image_width=640, image_height=480)
    pts = [
        TrajectoryPoint(x=0.0, y=0.0, z=1.0),   # visible
        TrajectoryPoint(x=0.0, y=0.0, z=-1.0),  # behind camera
    ]
    results = proj.project_trajectory(pts)
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is None


def test_extrinsic_translation_applied_correctly():
    # Camera is at z=1 in robot frame: T shifts z by -1
    T = np.eye(4)
    T[2, 3] = -1.0
    cam = CameraParams(fx=600.0, fy=600.0, cx=320.0, cy=240.0, T_cam_from_robot=T)
    proj = PinholeProjector(cam, image_width=640, image_height=480)
    # Robot point at z=2 → camera z=1 → should project to principal point
    assert proj.project(TrajectoryPoint(x=0.0, y=0.0, z=2.0)) == (320, 240)
