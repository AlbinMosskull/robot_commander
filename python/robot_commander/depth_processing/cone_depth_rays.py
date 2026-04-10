import math

import numpy as np

from robot_commander.image_processing.intrinsics import Intrinsics
from robot_commander.sensor.range_reading import RangeReading


def depth_to_rays(
    calibrated_depth: np.ndarray,
    intrinsics: Intrinsics,
    agent_x: float,
    agent_y: float,
    agent_heading: float,
    num_slices: int = 30,
    vertical_band_fraction: float = 0.2,
) -> list[RangeReading]:
    height, width = calibrated_depth.shape
    row_lo = int(height / 2 - height * vertical_band_fraction / 2)
    row_hi = int(height / 2 + height * vertical_band_fraction / 2)
    band = calibrated_depth[row_lo:row_hi, :]

    rays = []
    for i in range(num_slices):
        col_lo = i * width // num_slices
        col_hi = (i + 1) * width // num_slices
        slice_depths = band[:, col_lo:col_hi]
        valid = slice_depths[slice_depths > 0]
        if valid.size == 0:
            continue

        d = float(valid.min())
        col_center = (col_lo + col_hi) / 2
        theta_cam = math.atan2((col_center - intrinsics.cx) / intrinsics.fx, 1.0)
        horiz_dist = d * math.cos(theta_cam)
        theta_world = agent_heading + theta_cam
        end_x = agent_x + horiz_dist * math.cos(theta_world)
        end_y = agent_y + horiz_dist * math.sin(theta_world)
        rays.append(RangeReading(agent_x, agent_y, end_x, end_y, did_hit=True))

    return rays
