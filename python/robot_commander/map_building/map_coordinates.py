import numpy as np

_MAP_SCALE = 150
_MAP_W, _MAP_H = 600, 600
_MAP_ORIGIN = (300, 540)


def to_map_px(coords_2d: np.ndarray) -> np.ndarray:
    ox, oy = _MAP_ORIGIN
    return np.column_stack([
        (ox + coords_2d[:, 0] * _MAP_SCALE).astype(np.int32),
        (oy - coords_2d[:, 1] * _MAP_SCALE).astype(np.int32),
    ])


def px_to_world(px: int, py: int) -> tuple[float, float]:
    ox, oy = _MAP_ORIGIN
    return (px - ox) / _MAP_SCALE, (oy - py) / _MAP_SCALE


def world_to_px(x: float, y: float) -> tuple[int, int]:
    pts = to_map_px(np.array([[x, y]]))
    return int(pts[0, 0]), int(pts[0, 1])
