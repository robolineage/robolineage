import numpy as np
from robolineage_ar.types import CameraParams, RenderConfig, TrajectoryPoint


def test_trajectory_point_defaults():
    p = TrajectoryPoint(x=0.3, y=-0.1, z=0.5)
    assert p.x == 0.3
    assert p.color == (0, 0, 255)  # BGR red


def test_trajectory_point_custom_color():
    p = TrajectoryPoint(x=0.0, y=0.0, z=0.0, color=(255, 0, 0))
    assert p.color == (255, 0, 0)


def test_camera_params_default_extrinsics_is_identity():
    cam = CameraParams(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    assert cam.T_cam_from_robot.shape == (4, 4)
    np.testing.assert_array_equal(cam.T_cam_from_robot, np.eye(4))


def test_camera_params_custom_extrinsics():
    T = np.eye(4)
    T[2, 3] = 0.5
    cam = CameraParams(fx=600.0, fy=600.0, cx=320.0, cy=240.0, T_cam_from_robot=T)
    assert cam.T_cam_from_robot[2, 3] == 0.5


def test_render_config_defaults():
    cfg = RenderConfig()
    assert cfg.point_radius > 0
    assert cfg.line_thickness > 0
    assert cfg.trajectory_color == (0, 0, 255)
    assert cfg.max_points > 0
