from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_commander.image_processing.intrinsics import Intrinsics
from robot_commander.sensor.range_reading import RangeReading


@dataclass
class DepthCapture:
    frame: np.ndarray
    depth: np.ndarray
    cone_mask: np.ndarray
    ray_ends: np.ndarray  # (N, 2) float32, world-frame obstacle endpoints
    agent_x: float
    agent_y: float
    heading: float
    ultrasonic_min: float
    intrinsics: Intrinsics
    is_calibrated: bool = True


def save(capture: DepthCapture, path: Path) -> None:
    np.savez(
        path,
        frame=capture.frame,
        depth=capture.depth,
        cone_mask=capture.cone_mask,
        ray_ends=capture.ray_ends,
        agent_x=capture.agent_x,
        agent_y=capture.agent_y,
        heading=capture.heading,
        ultrasonic_min=capture.ultrasonic_min,
        camera_matrix=capture.intrinsics.camera_matrix,
        dist_coeffs=capture.intrinsics.dist_coeffs,
        intrinsics_rms_error=capture.intrinsics.rms_error,
        intrinsics_image_size=np.array(capture.intrinsics.image_size),
        is_calibrated=np.array(capture.is_calibrated),
    )


def load(path: Path) -> DepthCapture:
    data = np.load(path)
    intrinsics = Intrinsics(
        camera_matrix=data["camera_matrix"],
        dist_coeffs=data["dist_coeffs"],
        rms_error=float(data["intrinsics_rms_error"]),
        image_size=tuple(data["intrinsics_image_size"].tolist()),
    )
    depth_key = "depth" if "depth" in data else "calibrated_depth"
    is_calibrated = bool(data["is_calibrated"]) if "is_calibrated" in data else True
    return DepthCapture(
        frame=data["frame"],
        depth=data[depth_key],
        cone_mask=data["cone_mask"],
        ray_ends=data["ray_ends"],
        agent_x=float(data["agent_x"]),
        agent_y=float(data["agent_y"]),
        heading=float(data["heading"]),
        ultrasonic_min=float(data["ultrasonic_min"]),
        intrinsics=intrinsics,
        is_calibrated=is_calibrated,
    )


def rays_to_ends(rays: list[RangeReading]) -> np.ndarray:
    if not rays:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[r.end_x, r.end_y] for r in rays], dtype=np.float32)