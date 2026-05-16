import math

import numpy as np

from robot_commander.config import load as load_config
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud
from robot_commander.depth_processing.ransac import detect_floor
from robot_commander.image_processing.intrinsics import Intrinsics
from robot_commander.sensor.range_reading import RangeReading

_CAMERA_UP = np.array(load_config().depth.camera_up)
_MIN_OBSTACLE_HEIGHT_M = 0.05
_MIN_FLOOR_ALIGNMENT = 0.7
_BOTTOM_ROW_SKIP_FRACTION = 0.1


def floor_plane_basis(floor_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.array([1.0, 0.0, 0.0])
    right = right - (right @ floor_normal) * floor_normal
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, 0.0, 1.0])
        right = right - (right @ floor_normal) * floor_normal
    right /= np.linalg.norm(right)
    forward = np.cross(floor_normal, right)
    return right, forward


def _slices_to_rays(
    azimuths: np.ndarray,
    horizontal_distances: np.ndarray,
    slice_edges: np.ndarray,
    agent_x: float,
    agent_y: float,
    agent_heading: float,
) -> list[RangeReading]:
    rays = []
    for i in range(len(slice_edges) - 1):
        in_slice = (azimuths >= slice_edges[i]) & (azimuths < slice_edges[i + 1])
        if not in_slice.any():
            continue
        distance = float(np.percentile(horizontal_distances[in_slice], 5))
        slice_azimuth = (slice_edges[i] + slice_edges[i + 1]) / 2.0
        world_bearing = agent_heading - slice_azimuth
        end_x = agent_x + distance * math.cos(world_bearing)
        end_y = agent_y + distance * math.sin(world_bearing)
        rays.append(RangeReading(agent_x, agent_y, end_x, end_y, did_hit=True))
    return rays


def _rays_from_depth_direct(
    calibrated_depth: np.ndarray,
    intrinsics: Intrinsics,
    agent_x: float,
    agent_y: float,
    agent_heading: float,
    num_slices: int,
) -> list[RangeReading]:
    height, width = calibrated_depth.shape
    bottom_cutoff = height - int(height * _BOTTOM_ROW_SKIP_FRACTION)
    depth_trimmed = calibrated_depth[:bottom_cutoff, :]

    valid_mask = depth_trimmed > 0
    if not valid_mask.any():
        return []

    cols = np.where(valid_mask)[1].astype(np.float32)
    depths = depth_trimmed[valid_mask]
    x_camera = (cols - intrinsics.cx) * depths / intrinsics.fx
    azimuths = np.arctan2(x_camera, depths)

    half_fov = math.atan(width / (2.0 * intrinsics.fx))
    slice_edges = np.linspace(-half_fov, half_fov, num_slices + 1)

    return _slices_to_rays(azimuths, depths, slice_edges, agent_x, agent_y, agent_heading)


def depth_to_rays(
    calibrated_depth: np.ndarray,
    intrinsics: Intrinsics,
    agent_x: float,
    agent_y: float,
    agent_heading: float,
    robot_T_camera: np.ndarray = np.eye(4, dtype=np.float64),
    max_obstacle_height_m: float = 1.5,
    num_slices: int = 30,
) -> list[RangeReading]:
    camera_points = depth_image_to_point_cloud(calibrated_depth, intrinsics)
    ones = np.ones((len(camera_points), 1), dtype=np.float32)
    points = (np.hstack([camera_points, ones]) @ robot_T_camera.T)[:, :3]
    if len(points) < 3:
        return _rays_from_depth_direct(calibrated_depth, intrinsics, agent_x, agent_y, agent_heading, num_slices)

    floor = detect_floor(points)
    floor_alignment = abs(float(floor.normal @ _CAMERA_UP)) if floor is not None else 0.0
    if floor is None or floor_alignment < _MIN_FLOOR_ALIGNMENT:
        return _rays_from_depth_direct(calibrated_depth, intrinsics, agent_x, agent_y, agent_heading, num_slices)

    heights = points @ floor.normal - floor.distance
    valid = points[(heights > _MIN_OBSTACLE_HEIGHT_M) & (heights < max_obstacle_height_m)]
    if len(valid) == 0:
        return []

    right, forward = floor_plane_basis(floor.normal)
    rights = valid @ right
    forwards = valid @ forward
    azimuths = np.arctan2(rights, forwards)
    horizontal_distances = np.sqrt(rights ** 2 + forwards ** 2)

    width = calibrated_depth.shape[1]
    half_fov = math.atan(width / (2.0 * intrinsics.fx))
    slice_edges = np.linspace(-half_fov, half_fov, num_slices + 1)

    return _slices_to_rays(azimuths, horizontal_distances, slice_edges, agent_x, agent_y, agent_heading)
