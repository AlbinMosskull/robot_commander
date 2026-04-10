import math

import numpy as np
import pytest

from robot_commander.image_processing.intrinsics import Intrinsics
from robot_commander.depth_processing.cone_depth_rays import depth_to_rays


def _make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> Intrinsics:
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return Intrinsics(camera_matrix=K, dist_coeffs=np.zeros(5), rms_error=0.0, image_size=(0, 0))


def _uniform_depth(height: int, width: int, value: float) -> np.ndarray:
    return np.full((height, width), value, dtype=np.float32)


def test_forward_ray_at_center_column():
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
    intrinsics = _make_intrinsics(fx, fy, cx, cy)
    depth = _uniform_depth(480, 640, 1.0)

    # heading = π/2 means agent faces along +y axis
    # center column aligns with principal point → theta_cam ≈ 0 → ray goes straight forward
    rays = depth_to_rays(depth, intrinsics, agent_x=0.0, agent_y=0.0, agent_heading=math.pi / 2, num_slices=1)

    assert len(rays) == 1
    ray = rays[0]
    assert ray.start_x == pytest.approx(0.0)
    assert ray.start_y == pytest.approx(0.0)
    assert ray.end_x == pytest.approx(0.0, abs=1e-3)
    assert ray.end_y == pytest.approx(1.0, abs=1e-3)
    assert ray.did_hit is True


def test_off_center_column_produces_correct_angle():
    fx = 500.0
    cx = 320.0
    intrinsics = _make_intrinsics(fx, 500.0, cx, 240.0)
    depth = _uniform_depth(480, 640, 1.0)

    # Use a single slice covering the full width so col_center = cx + offset
    # Place agent at origin facing +x (heading=0)
    # With heading=0: theta_world = 0 + theta_cam
    # col_center for the only slice = (0 + 640) / 2 = 320 = cx → theta_cam = 0
    rays = depth_to_rays(depth, intrinsics, agent_x=0.0, agent_y=0.0, agent_heading=0.0, num_slices=2)

    assert len(rays) == 2

    # Left slice: col_center = 160, theta_cam = atan2((160 - 320)/500, 1) < 0
    left_ray = rays[0]
    right_ray = rays[1]

    left_col_center = 160.0
    right_col_center = 480.0
    theta_left = math.atan2((left_col_center - cx) / fx, 1.0)
    theta_right = math.atan2((right_col_center - cx) / fx, 1.0)

    # heading=0 means agent faces +x, so end_x = horiz_dist * cos(theta_cam), end_y = horiz_dist * sin(theta_cam)
    assert left_ray.end_y == pytest.approx(math.cos(theta_left) * math.sin(theta_left), abs=1e-3)
    assert right_ray.end_x == pytest.approx(math.cos(theta_right) * math.cos(theta_right), abs=1e-3)


def test_minimum_depth_used_per_slice():
    intrinsics = _make_intrinsics(500.0, 500.0, 320.0, 240.0)
    height, width = 480, 640
    depth = _uniform_depth(height, width, 5.0)

    # Place a closer pixel in the vertical band at the center column
    row_mid = height // 2
    depth[row_mid, width // 2] = 2.0

    rays = depth_to_rays(depth, intrinsics, agent_x=0.0, agent_y=0.0, agent_heading=math.pi / 2, num_slices=1)

    assert len(rays) == 1
    # The minimum depth in the single slice is 2.0, not 5.0
    ray = rays[0]
    # end_y ≈ 2.0 (forward direction with heading=π/2, center column)
    assert ray.end_y == pytest.approx(2.0, abs=0.1)
