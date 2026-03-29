import math

_RAY_RANGE = 1.0
_N_RAYS = 21
_SWEEP_DEG = 120.0


def cast_rays(x: float, y: float, heading: float) -> list[tuple[float, float, float, float]]:
    half_sweep = math.radians(_SWEEP_DEG / 2)
    rays = []
    for i in range(_N_RAYS):
        t = i / (_N_RAYS - 1)
        angle = heading + (-half_sweep + t * 2 * half_sweep)
        end_x = x + _RAY_RANGE * math.cos(angle)
        end_y = y + _RAY_RANGE * math.sin(angle)
        rays.append((x, y, end_x, end_y))
    return rays
