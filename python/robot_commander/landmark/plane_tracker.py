import threading

import numpy as np

from robot_commander.landmark.landmark_plane import LandmarkPlane

_CENTERPOINT_MATCH_M = 0.2


class PlaneTracker:
    def __init__(self) -> None:
        self._planes: list[LandmarkPlane] = []
        self._lock = threading.Lock()

    def update(self, observations: list[LandmarkPlane], is_calibrated: bool) -> None:
        with self._lock:
            for obs in observations:
                match_idx = self._find_match_index(obs)
                if match_idx is None:
                    self._planes.append(_with_locked(obs, is_calibrated))
                else:
                    self._planes[match_idx] = _merge(self._planes[match_idx], obs, is_calibrated)

    def planes(self) -> list[LandmarkPlane]:
        with self._lock:
            return list(self._planes)

    def _find_match_index(self, obs: LandmarkPlane) -> int | None:
        obs_center = (obs.endpoint_a + obs.endpoint_b) / 2
        for i, plane in enumerate(self._planes):
            plane_center = (plane.endpoint_a + plane.endpoint_b) / 2
            if float(np.linalg.norm(obs_center - plane_center)) < _CENTERPOINT_MATCH_M:
                return i
        return None


def _with_locked(plane: LandmarkPlane, is_calibrated: bool) -> LandmarkPlane:
    if not is_calibrated:
        return plane
    return LandmarkPlane(
        normal=plane.normal,
        distance=plane.distance,
        endpoint_a=plane.endpoint_a,
        endpoint_b=plane.endpoint_b,
        is_locked=True,
        observation_count=plane.observation_count,
    )


def _merge(existing: LandmarkPlane, obs: LandmarkPlane, is_calibrated: bool) -> LandmarkPlane:
    if float(obs.normal @ existing.normal) < 0:
        obs = LandmarkPlane(
            normal=-obs.normal,
            distance=-obs.distance,
            endpoint_a=obs.endpoint_a,
            endpoint_b=obs.endpoint_b,
            is_locked=obs.is_locked,
            observation_count=obs.observation_count,
        )
    count = existing.observation_count + 1

    if existing.is_locked:
        merged_normal = existing.normal
        merged_distance = existing.distance
        merged_locked = True
    else:
        weight = 1.0 / count
        raw_normal = (1.0 - weight) * existing.normal + weight * obs.normal
        magnitude = float(np.linalg.norm(raw_normal))
        merged_normal = raw_normal / magnitude if magnitude > 1e-6 else existing.normal
        merged_distance = (1.0 - weight) * existing.distance + weight * obs.distance
        merged_locked = is_calibrated

    tangent = np.array([-merged_normal[1], merged_normal[0]])
    if merged_locked and existing.is_locked:
        all_ends = np.stack([existing.endpoint_a, existing.endpoint_b, obs.endpoint_a, obs.endpoint_b])
        projs = all_ends @ tangent
    else:
        projs = np.array([obs.endpoint_a @ tangent, obs.endpoint_b @ tangent])
    endpoint_a = merged_distance * merged_normal + float(projs.min()) * tangent
    endpoint_b = merged_distance * merged_normal + float(projs.max()) * tangent

    return LandmarkPlane(
        normal=merged_normal,
        distance=merged_distance,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        is_locked=merged_locked,
        observation_count=count,
    )
