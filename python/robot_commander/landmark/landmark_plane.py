from dataclasses import dataclass

import numpy as np


@dataclass
class LandmarkPlane:
    normal: np.ndarray      # 2D unit normal in world XY, shape (2,)
    distance: float         # signed distance from world origin: p @ normal = distance
    endpoint_a: np.ndarray  # world 2D endpoint, shape (2,)
    endpoint_b: np.ndarray  # world 2D endpoint, shape (2,)
    is_locked: bool = False
    observation_count: int = 1
