import math
from pathlib import Path

import cv2
import numpy as np

from robot_commander.map_building.map_coordinates import world_to_px, _MAP_W, _MAP_H

_RAY_RANGE = 1.0
_RAY_STEPS = 60
_SWEEP_DEG = 120.0
_SWEEP_DEG_PER_SEC = 60.0

_OBSTACLES_PATH = Path(__file__).parent / "obstacles.png"

_obstacles: np.ndarray | None = None
_obstacles_loaded = False


def _get_obstacles() -> np.ndarray | None:
    global _obstacles, _obstacles_loaded
    if not _obstacles_loaded:
        _obstacles_loaded = True
        if _OBSTACLES_PATH.exists():
            _obstacles = cv2.imread(str(_OBSTACLES_PATH), cv2.IMREAD_GRAYSCALE)
    return _obstacles


def cast_ray(x: float, y: float, angle: float) -> tuple[float, float, float, float, bool]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    obstacles = _get_obstacles()

    for i in range(1, _RAY_STEPS + 1):
        t = i / _RAY_STEPS * _RAY_RANGE
        cx = x + t * cos_a
        cy = y + t * sin_a

        if obstacles is not None:
            px, py = world_to_px(cx, cy)
            if 0 <= px < _MAP_W and 0 <= py < _MAP_H and obstacles[py, px] > 0:
                return (x, y, cx, cy, True)

    return (x, y, x + _RAY_RANGE * cos_a, y + _RAY_RANGE * sin_a, False)
