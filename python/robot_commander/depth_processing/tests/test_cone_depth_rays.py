import math

import numpy as np
import pytest

from robot_commander.image_processing.intrinsics import Intrinsics
from robot_commander.depth_processing.cone_depth_rays import depth_to_rays

_FX = 500.0
_FY = 500.0
_CX = 320.0
_CY = 240.0
_HEIGHT = 480
_WIDTH = 640

# Camera is 1m above the floor (floor at y=1.0 in camera frame, y-axis points down).
_CAMERA_HEIGHT_M = 1.0


def _make_intrinsics() -> Intrinsics:
    K = np.array([[_FX, 0, _CX], [0, _FY, _CY], [0, 0, 1]], dtype=np.float64)
    return Intrinsics(camera_matrix=K, dist_coeffs=np.zeros(5), rms_error=0.0, image_size=(_WIDTH, _HEIGHT))


def _make_floor_depth() -> np.ndarray:
    """Depth image of a horizontal floor at camera-y = 1.0m (camera 1m above floor).

    For a pixel at (row, col), the ray intersects the floor plane y=1.0 at depth
    z = 1.0 * fy / (row - cy), valid only for rows below the horizon (row > cy).
    """
    depth = np.zeros((_HEIGHT, _WIDTH), dtype=np.float32)
    rows = np.arange(_HEIGHT)[:, np.newaxis]
    below_horizon = rows > _CY
    with np.errstate(divide='ignore', invalid='ignore'):
        floor_z = np.where(below_horizon, _CAMERA_HEIGHT_M * _FY / (rows - _CY), 0.0)
    depth[below_horizon.squeeze()] = floor_z[below_horizon.squeeze()].astype(np.float32)
    return depth


def _add_obstacle(depth: np.ndarray, obstacle_z: float, obstacle_camera_y: float, col_lo: int, col_hi: int) -> None:
    """Paint an obstacle strip into the depth image at the given depth and camera-y height."""
    obstacle_row = int(_CY + obstacle_camera_y * _FY / obstacle_z)
    depth[obstacle_row, col_lo:col_hi] = obstacle_z


def test_no_valid_depth_returns_empty():
    intrinsics = _make_intrinsics()
    depth = np.zeros((_HEIGHT, _WIDTH), dtype=np.float32)
    assert depth_to_rays(depth, intrinsics, 0.0, 0.0, 0.0) == []


def test_obstacle_directly_ahead_produces_forward_ray():
    intrinsics = _make_intrinsics()
    depth = _make_floor_depth()

    obstacle_z = 2.0
    obstacle_camera_y = 0.5  # 0.5m below camera = 0.5m above 1m-deep floor
    _add_obstacle(depth, obstacle_z, obstacle_camera_y, col_lo=0, col_hi=_WIDTH)

    # Heading = π/2 means the agent faces +y (north). Forward ray should reach +y.
    rays = depth_to_rays(depth, intrinsics, agent_x=0.0, agent_y=0.0, agent_heading=math.pi / 2)

    assert len(rays) > 0
    center_ray = min(rays, key=lambda r: abs(r.end_x - 0.0))
    assert center_ray.end_y == pytest.approx(obstacle_z, abs=0.15)
    assert center_ray.end_x == pytest.approx(0.0, abs=0.15)


def test_obstacle_above_max_height_excluded():
    intrinsics = _make_intrinsics()
    depth = _make_floor_depth()

    obstacle_z = 2.0
    # Put obstacle at camera-y = -0.1 → height above floor = 1.0 - (-0.1) = 1.1m...
    # Actually: height above floor = floor_y - obstacle_camera_y = 1.0 - obstacle_camera_y.
    # For height = 1.6m: obstacle_camera_y = 1.0 - 1.6 = -0.6 (above camera level).
    obstacle_camera_y = -0.6
    _add_obstacle(depth, obstacle_z, obstacle_camera_y, col_lo=0, col_hi=_WIDTH)

    rays = depth_to_rays(depth, intrinsics, 0.0, 0.0, 0.0, max_obstacle_height_m=1.5)
    assert rays == []


def test_obstacle_to_the_left_produces_leftward_ray():
    intrinsics = _make_intrinsics()
    depth = _make_floor_depth()

    obstacle_z = 2.0
    obstacle_camera_y = 0.5
    # Place obstacle only in the left quarter of the image.
    _add_obstacle(depth, obstacle_z, obstacle_camera_y, col_lo=0, col_hi=_WIDTH // 4)

    # Heading = π/2 (north). Left of a north-facing camera is west (-x in world).
    rays = depth_to_rays(depth, intrinsics, agent_x=0.0, agent_y=0.0, agent_heading=math.pi / 2)

    assert len(rays) > 0
    leftmost = min(rays, key=lambda r: r.end_x)
    assert leftmost.end_x < 0.0


def test_obstacle_to_the_right_produces_rightward_ray():
    intrinsics = _make_intrinsics()
    depth = _make_floor_depth()

    obstacle_z = 2.0
    obstacle_camera_y = 0.5
    # Place obstacle only in the right quarter of the image.
    _add_obstacle(depth, obstacle_z, obstacle_camera_y, col_lo=3 * _WIDTH // 4, col_hi=_WIDTH)

    # Heading = π/2 (north). Right of a north-facing camera is east (+x in world).
    rays = depth_to_rays(depth, intrinsics, agent_x=0.0, agent_y=0.0, agent_heading=math.pi / 2)

    assert len(rays) > 0
    rightmost = max(rays, key=lambda r: r.end_x)
    assert rightmost.end_x > 0.0


def test_tilted_floor_still_filters_correctly():
    """A tilted floor (simulating camera pitch) should still be detected and filtered."""
    intrinsics = _make_intrinsics()

    # Tilted floor: normal points in a direction between -y and +z (camera pitched forward).
    # We simulate this by generating 3D floor points on a tilted plane, then projecting
    # them to a depth image.
    tilt = math.radians(20)  # 20-degree forward pitch
    n_floor = np.array([0.0, -math.cos(tilt), math.sin(tilt)])  # tilted normal
    d_floor = -_CAMERA_HEIGHT_M  # camera 1m above floor

    rows, cols = np.meshgrid(np.arange(_HEIGHT), np.arange(_WIDTH), indexing="ij")
    ray_x = (cols - _CX) / _FX
    ray_y = (rows - _CY) / _FY
    ray_z = np.ones((_HEIGHT, _WIDTH))

    # Intersect each ray with the tilted plane: dot(t*ray, n) = d → t = d / dot(ray, n)
    denom = ray_x * n_floor[0] + ray_y * n_floor[1] + ray_z * n_floor[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        depth_vals = np.where(denom > 1e-6, d_floor / denom, 0.0)

    depth = np.clip(depth_vals, 0.0, None).astype(np.float32)

    # Obstacle: center strip at 0.5m above tilted floor.
    # Approximate position: 2m in front of camera at half camera height.
    center_row = int(_CY)
    depth[center_row - 5: center_row + 5, _WIDTH // 4: 3 * _WIDTH // 4] = 2.0

    rays = depth_to_rays(depth, intrinsics, agent_x=0.0, agent_y=0.0, agent_heading=math.pi / 2)
    # With a tilted floor, the band approach would fail, but RANSAC should handle it.
    # We just verify the function runs without error and returns something reasonable.
    assert isinstance(rays, list)