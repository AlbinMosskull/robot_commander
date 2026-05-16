import math
from dataclasses import dataclass

import numpy as np

from robot_commander.depth_processing.cone_depth_processor import ConeDepthProcessor
from robot_commander.depth_processing.cone_depth_rays import depth_to_rays
from robot_commander.depth_processing.depth_planes import (
    LandmarkPlaneDebugResult,
    extract_landmark_planes_debug,
)
from robot_commander.depth_processing.ultrasonic_plane_validator import PlaneValidationResult
from robot_commander.image_processing.intrinsics import Intrinsics
from robot_commander.landmark.landmark_plane import LandmarkPlane
from robot_commander.sensor.range_reading import RangeReading

_DEPTH_RAY_RANGE_FACTOR = 2.5


@dataclass
class DepthFrameInput:
    frame: np.ndarray
    ultrasonic_min: float
    agent_x: float
    agent_y: float
    agent_heading: float


@dataclass
class OccupancyRays:
    gaussian_hit_rays: list[tuple[float, float, float, float]]
    free_rays: list[tuple[float, float, float, float]]


@dataclass
class DepthFrameResult:
    occupancy_rays: OccupancyRays
    landmark_plane_observations: list[LandmarkPlane]
    is_calibrated: bool

    frame: np.ndarray
    raw_depth: np.ndarray
    calibrated_depth: np.ndarray
    cone_mask: np.ndarray
    agent_x: float
    agent_y: float
    agent_heading: float
    ultrasonic_min: float
    intrinsics: Intrinsics

    validation: PlaneValidationResult
    validation_mask: np.ndarray
    landmark_debug: LandmarkPlaneDebugResult
    depth_rays: list[RangeReading]


def process_depth_frame(
    depth_input: DepthFrameInput,
    processor: ConeDepthProcessor,
    intrinsics: Intrinsics,
) -> DepthFrameResult:
    raw_depth, calibrated_depth, cone_mask, validation_mask, validation = (
        processor.process_with_validation(depth_input.frame, depth_input.ultrasonic_min)
    )
    is_calibrated = validation.disqualification_reason is None
    depth_for_update = calibrated_depth if is_calibrated else raw_depth
    max_ray_m = _DEPTH_RAY_RANGE_FACTOR * depth_input.ultrasonic_min

    depth_rays = depth_to_rays(
        depth_for_update, intrinsics,
        depth_input.agent_x, depth_input.agent_y, depth_input.agent_heading,
    )
    occupancy_rays = (
        _calibrated_occupancy_rays(depth_rays, max_ray_m)
        if is_calibrated
        else _conservative_occupancy_rays(depth_rays, max_ray_m)
    )
    landmark_debug = extract_landmark_planes_debug(
        depth_for_update, intrinsics,
        depth_input.agent_x, depth_input.agent_y, depth_input.agent_heading,
    )

    return DepthFrameResult(
        occupancy_rays=occupancy_rays,
        landmark_plane_observations=[
            info.landmark for info in landmark_debug.planes if info.landmark is not None
        ],
        is_calibrated=is_calibrated,
        frame=depth_input.frame,
        raw_depth=raw_depth,
        calibrated_depth=calibrated_depth,
        cone_mask=cone_mask,
        agent_x=depth_input.agent_x,
        agent_y=depth_input.agent_y,
        agent_heading=depth_input.agent_heading,
        ultrasonic_min=depth_input.ultrasonic_min,
        intrinsics=intrinsics,
        validation=validation,
        validation_mask=validation_mask,
        landmark_debug=landmark_debug,
        depth_rays=depth_rays,
    )


def _calibrated_occupancy_rays(rays: list[RangeReading], max_ray_m: float) -> OccupancyRays:
    gaussian_hits = []
    free = []
    for ray in rays:
        sx, sy, ex, ey, did_hit = _clip_ray(ray, max_ray_m)
        if did_hit:
            gaussian_hits.append((sx, sy, ex, ey))
        else:
            free.append((sx, sy, ex, ey))
    return OccupancyRays(gaussian_hit_rays=gaussian_hits, free_rays=free)


def _conservative_occupancy_rays(rays: list[RangeReading], max_ray_m: float) -> OccupancyRays:
    return OccupancyRays(
        gaussian_hit_rays=[],
        free_rays=[_halve_ray(ray, max_ray_m) for ray in rays],
    )


def _clip_ray(ray: RangeReading, max_length_m: float) -> tuple[float, float, float, float, bool]:
    dx = ray.end_x - ray.start_x
    dy = ray.end_y - ray.start_y
    length = math.sqrt(dx ** 2 + dy ** 2)
    if length <= max_length_m:
        return ray.start_x, ray.start_y, ray.end_x, ray.end_y, ray.did_hit
    scale = max_length_m / length
    return ray.start_x, ray.start_y, ray.start_x + dx * scale, ray.start_y + dy * scale, False


def _halve_ray(ray: RangeReading, max_length_m: float) -> tuple[float, float, float, float]:
    sx, sy, ex, ey, _ = _clip_ray(ray, max_length_m)
    return sx, sy, sx + (ex - sx) * 0.5, sy + (ey - sy) * 0.5
