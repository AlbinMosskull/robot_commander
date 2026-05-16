import math
from dataclasses import dataclass

import numpy as np

from robot_commander.depth_processing.cone_depth_rays import floor_plane_basis, _CAMERA_UP
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud
from robot_commander.depth_processing.ransac import Plane, detect_planes
from robot_commander.image_processing.intrinsics import Intrinsics


def _camera_to_world_2d(
    points: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    agent_x: float,
    agent_y: float,
    agent_heading: float,
) -> np.ndarray:
    rights = points @ right
    forwards = points @ forward
    azimuths = np.arctan2(rights, forwards)
    horiz_dists = np.sqrt(rights ** 2 + forwards ** 2)
    bearings = agent_heading - azimuths
    return np.stack([
        agent_x + horiz_dists * np.cos(bearings),
        agent_y + horiz_dists * np.sin(bearings),
    ], axis=-1)


def _largest_cluster(world_2d: np.ndarray, tangent_projs: np.ndarray, max_gap_m: float) -> np.ndarray | None:
    order = np.argsort(tangent_projs)
    sorted_projs = tangent_projs[order]
    split_indices = np.where(np.diff(sorted_projs) > max_gap_m)[0] + 1
    clusters = np.split(order, split_indices)
    if not clusters:
        return None
    best = max(clusters, key=len)
    return world_2d[best]


def _camera_normal_to_world_2d(
    camera_normal: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    agent_heading: float,
) -> np.ndarray | None:
    n_right = float(camera_normal @ right)
    n_forward = float(camera_normal @ forward)
    magnitude = math.sqrt(n_right ** 2 + n_forward ** 2)
    if magnitude < 1e-6:
        return None
    azimuth = math.atan2(n_right, n_forward)
    bearing = agent_heading - azimuth
    return np.array([math.cos(bearing), math.sin(bearing)])
