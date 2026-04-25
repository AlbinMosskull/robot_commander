from dataclasses import dataclass
import math

import numpy as np

from robot_commander.depth_processing.ransac import Plane, detect_planes

_SENSOR_FORWARD = np.array([0.0, 0.0, 1.0])
_N_PLANES = 3
_RANSAC_ITERATIONS = 100
_RANSAC_DISTANCE_THRESHOLD_M = 0.03
_MIN_CONE_FILL_FRACTION = 0.10
_MAX_NORMAL_ANGLE_DEG = 20


@dataclass(frozen=True)
class PlaneCandidate:
    plane: Plane
    cone_fill_fraction: float
    normal_angle_deg: float


@dataclass(frozen=True)
class PlaneValidationResult:
    best_candidate: PlaneCandidate | None
    all_candidates: list[PlaneCandidate]
    disqualification_reason: str | None


def validate_ultrasonic_with_planes(
    cone_points: np.ndarray,
    min_cone_fill_fraction: float = _MIN_CONE_FILL_FRACTION,
    max_normal_angle_deg: float = _MAX_NORMAL_ANGLE_DEG,
) -> PlaneValidationResult:
    if len(cone_points) < 3:
        return PlaneValidationResult(
            best_candidate=None,
            all_candidates=[],
            disqualification_reason="no planes detected",
        )

    planes = detect_planes(
        cone_points,
        n_planes=_N_PLANES,
        n_iterations=_RANSAC_ITERATIONS,
        distance_threshold=_RANSAC_DISTANCE_THRESHOLD_M,
    )

    if not planes:
        return PlaneValidationResult(
            best_candidate=None,
            all_candidates=[],
            disqualification_reason="no planes detected",
        )

    candidates = [_make_candidate(plane, len(cone_points)) for plane in planes]
    candidates.sort(key=lambda c: c.cone_fill_fraction, reverse=True)
    best = candidates[0]

    if best.cone_fill_fraction < min_cone_fill_fraction:
        reason = "plane covers too small a fraction of the cone"
    elif best.normal_angle_deg > max_normal_angle_deg:
        reason = "plane normal too far from sensor direction"
    else:
        reason = None

    return PlaneValidationResult(
        best_candidate=best,
        all_candidates=candidates,
        disqualification_reason=reason,
    )


def _make_candidate(plane: Plane, total_cone_points: int) -> PlaneCandidate:
    cone_fill_fraction = float(plane.inliers.sum()) / total_cone_points
    cos_angle = float(abs(np.dot(plane.normal, _SENSOR_FORWARD)))
    cos_angle = min(1.0, cos_angle)
    normal_angle_deg = math.degrees(math.acos(cos_angle))
    return PlaneCandidate(
        plane=plane,
        cone_fill_fraction=cone_fill_fraction,
        normal_angle_deg=normal_angle_deg,
    )
