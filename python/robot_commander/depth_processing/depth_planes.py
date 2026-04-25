import math
from dataclasses import dataclass

import numpy as np

from robot_commander.depth_processing.cone_depth_rays import floor_plane_basis, _CAMERA_UP
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud
from robot_commander.depth_processing.ransac import Plane, detect_planes
from robot_commander.image_processing.intrinsics import Intrinsics
from robot_commander.landmark.landmark_plane import LandmarkPlane

_N_PLANES = 4
_RANSAC_ITERATIONS = 100
_RANSAC_DISTANCE_THRESHOLD_M = 0.03
_FLOOR_DOT_THRESHOLD = 0.6
_MIN_INLIERS = 50
_MIN_INLIER_FRACTION = 0.03
_MAX_GAP_M = 0.3
_MIN_EXTENT_M = 0.2
_MAX_EXTENT_M = 1.0
_MAX_DEPTH_M = 3.0
_MIN_FLOOR_ALIGNMENT = 0.7


@dataclass
class PlaneDebugInfo:
    world_2d_points: np.ndarray   # (N, 2) inlier points projected to world XY
    landmark: LandmarkPlane | None
    rejection_reason: str | None  # None when accepted


@dataclass
class LandmarkPlaneDebugResult:
    planes: list[PlaneDebugInfo]
    floor_alignment: float | None  # dot product of best floor candidate with _CAMERA_UP; None if no planes found


def extract_landmark_planes(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    agent_x: float,
    agent_y: float,
    agent_heading: float,
) -> list[LandmarkPlane]:
    result = extract_landmark_planes_debug(depth, intrinsics, agent_x, agent_y, agent_heading)
    return [info.landmark for info in result.planes if info.landmark is not None]


def extract_landmark_planes_debug(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    agent_x: float,
    agent_y: float,
    agent_heading: float,
) -> LandmarkPlaneDebugResult:
    camera_points = depth_image_to_point_cloud(depth, intrinsics)
    camera_points = camera_points[camera_points[:, 2] <= _MAX_DEPTH_M]
    if len(camera_points) < _MIN_INLIERS:
        return LandmarkPlaneDebugResult(planes=[], floor_alignment=None)

    all_planes = detect_planes(
        camera_points,
        n_planes=_N_PLANES,
        n_iterations=_RANSAC_ITERATIONS,
        distance_threshold=_RANSAC_DISTANCE_THRESHOLD_M,
    )
    if not all_planes:
        return LandmarkPlaneDebugResult(planes=[], floor_alignment=None)

    floor_plane = max(all_planes, key=lambda p: abs(float(p.normal @ _CAMERA_UP)))
    floor_alignment = abs(float(floor_plane.normal @ _CAMERA_UP))

    print(f"  [planes] agent=({agent_x:.3f},{agent_y:.3f}) heading={math.degrees(agent_heading):.1f}°")
    print(f"  [planes] floor_normal={floor_plane.normal} alignment={floor_alignment:.3f}")

    if floor_alignment < _MIN_FLOOR_ALIGNMENT:
        return LandmarkPlaneDebugResult(planes=[], floor_alignment=floor_alignment)

    if float(floor_plane.normal @ _CAMERA_UP) < 0:
        floor_plane = Plane(-floor_plane.normal, -floor_plane.distance, floor_plane.inliers)
    right, forward = floor_plane_basis(floor_plane.normal)
    print(f"  [planes] right={right}  forward={forward}")

    debug_infos: list[PlaneDebugInfo] = []
    for idx, plane in enumerate(all_planes):
        if plane is floor_plane:
            continue
        inlier_points = camera_points[plane.inliers]
        world_2d = _camera_to_world_2d(inlier_points, right, forward, agent_x, agent_y, agent_heading)
        landmark, reason = _to_landmark_with_reason(plane, camera_points, floor_plane.normal, right, forward, world_2d, agent_heading, plane_idx=idx)
        debug_infos.append(PlaneDebugInfo(world_2d_points=world_2d, landmark=landmark, rejection_reason=reason))

    return LandmarkPlaneDebugResult(planes=debug_infos, floor_alignment=floor_alignment)


def _to_landmark_with_reason(
    plane: Plane,
    all_points: np.ndarray,
    floor_normal: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    world_2d: np.ndarray,
    agent_heading: float,
    plane_idx: int = -1,
) -> tuple[LandmarkPlane | None, str | None]:
    floor_dot = abs(float(plane.normal @ floor_normal))
    if floor_dot > _FLOOR_DOT_THRESHOLD:
        print(f"  [plane {plane_idx}] REJECTED floor-like: cam_normal={plane.normal} dot={floor_dot:.2f}")
        return None, f"floor-like (dot={floor_dot:.2f})"

    inlier_count = int(plane.inliers.sum())
    if inlier_count < _MIN_INLIERS:
        print(f"  [plane {plane_idx}] REJECTED too few inliers: {inlier_count}")
        return None, f"too few inliers ({inlier_count})"
    if inlier_count / len(all_points) < _MIN_INLIER_FRACTION:
        print(f"  [plane {plane_idx}] REJECTED fraction too small: {inlier_count/len(all_points):.3f}")
        return None, f"inlier fraction too small ({inlier_count / len(all_points):.3f})"

    world_normal = _camera_normal_to_world_2d(plane.normal, right, forward, agent_heading)
    if world_normal is None:
        return None, "normal has no horizontal component"

    print(f"  [plane {plane_idx}] cam_normal={plane.normal}  world_normal={world_normal}")

    tangent = np.array([-world_normal[1], world_normal[0]])
    tangent_projs = world_2d @ tangent
    cluster_points = _largest_cluster(world_2d, tangent_projs, _MAX_GAP_M)

    if cluster_points is None:
        return None, "no cluster found"

    cluster_tangent_projs = cluster_points @ tangent
    extent = float(cluster_tangent_projs.max() - cluster_tangent_projs.min())
    if extent < _MIN_EXTENT_M:
        print(f"  [plane {plane_idx}] REJECTED cluster too small: extent={extent:.3f}m")
        return None, f"largest cluster extent too small ({extent:.2f} m)"
    if extent > _MAX_EXTENT_M:
        print(f"  [plane {plane_idx}] REJECTED cluster too large: extent={extent:.3f}m")
        return None, f"largest cluster extent too large ({extent:.2f} m)"

    centroid = cluster_points.mean(axis=0)
    world_distance = float(centroid @ world_normal)
    endpoint_a = world_distance * world_normal + float(cluster_tangent_projs.min()) * tangent
    endpoint_b = world_distance * world_normal + float(cluster_tangent_projs.max()) * tangent

    print(f"  [plane {plane_idx}] ACCEPTED: world_dist={world_distance:.3f}m  extent={extent:.3f}m  "
          f"A={endpoint_a}  B={endpoint_b}  inliers={inlier_count}")
    print(f"  [plane {plane_idx}]   world_2d range x=[{world_2d[:,0].min():.3f},{world_2d[:,0].max():.3f}] "
          f"y=[{world_2d[:,1].min():.3f},{world_2d[:,1].max():.3f}]")
    print(f"  [plane {plane_idx}]   cam_inlier depth range [{plane.distance - _RANSAC_DISTANCE_THRESHOLD_M:.3f}, {plane.distance + _RANSAC_DISTANCE_THRESHOLD_M:.3f}]m  "
          f"cam_dist={plane.distance:.3f}")

    return LandmarkPlane(
        normal=world_normal,
        distance=world_distance,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
    ), None


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
