from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_commander.sensor.range_reading import RangeReading


@dataclass
class DepthCapture:
    frame: np.ndarray
    calibrated_depth: np.ndarray
    cone_mask: np.ndarray
    ray_ends: np.ndarray  # (N, 2) float32, world-frame obstacle endpoints
    agent_x: float
    agent_y: float
    heading: float
    ultrasonic_min: float


def save(capture: DepthCapture, path: Path) -> None:
    np.savez(
        path,
        frame=capture.frame,
        calibrated_depth=capture.calibrated_depth,
        cone_mask=capture.cone_mask,
        ray_ends=capture.ray_ends,
        agent_x=capture.agent_x,
        agent_y=capture.agent_y,
        heading=capture.heading,
        ultrasonic_min=capture.ultrasonic_min,
    )


def load(path: Path) -> DepthCapture:
    data = np.load(path)
    return DepthCapture(
        frame=data["frame"],
        calibrated_depth=data["calibrated_depth"],
        cone_mask=data["cone_mask"],
        ray_ends=data["ray_ends"],
        agent_x=float(data["agent_x"]),
        agent_y=float(data["agent_y"]),
        heading=float(data["heading"]),
        ultrasonic_min=float(data["ultrasonic_min"]),
    )


def rays_to_ends(rays: list[RangeReading]) -> np.ndarray:
    if not rays:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[r.end_x, r.end_y] for r in rays], dtype=np.float32)
