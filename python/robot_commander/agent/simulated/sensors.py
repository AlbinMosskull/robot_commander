import math
from abc import ABC, abstractmethod

from robot_commander.agent.data_types import RangeReading
from robot_commander.agent.simulated.ray_caster import cast_ray, _RAY_RANGE, _SWEEP_DEG, _SWEEP_DEG_PER_SEC

_TICK_HZ = 10

_CONE_HALF_ANGLE_DEG = 15.0
_CONE_RAYS = 9


class SimulatedSensor(ABC):
    @abstractmethod
    def read(self, x: float, y: float, heading: float) -> list[RangeReading]: ...


class SweepSensor(SimulatedSensor):
    """Single ray oscillating across a wide arc."""

    def __init__(self):
        self._sweep_offset: float = -math.radians(_SWEEP_DEG / 2)
        self._sweep_dir: float = 1.0

    def read(self, x: float, y: float, heading: float) -> list[RangeReading]:
        step = math.radians(_SWEEP_DEG_PER_SEC / _TICK_HZ)
        half = math.radians(_SWEEP_DEG / 2)
        self._sweep_offset += self._sweep_dir * step
        if self._sweep_offset >= half:
            self._sweep_offset = half
            self._sweep_dir = -1.0
        elif self._sweep_offset <= -half:
            self._sweep_offset = -half
            self._sweep_dir = 1.0
        sx, sy, ex, ey, did_hit = cast_ray(x, y, heading + self._sweep_offset)
        return [RangeReading(sx, sy, ex, ey, did_hit)]


class ConeSensor(SimulatedSensor):
    """Casts several rays across a cone and returns the closest hit as a
    single reading, mimicking a sensor that reports one distance value."""

    def __init__(
        self,
        half_angle_deg: float = _CONE_HALF_ANGLE_DEG,
        num_rays: int = _CONE_RAYS,
    ):
        self._half_angle = math.radians(half_angle_deg)
        self._num_rays = num_rays

    def read(self, x: float, y: float, heading: float) -> list[RangeReading]:
        angles = [
            heading + self._half_angle * (2 * i / (self._num_rays - 1) - 1)
            for i in range(self._num_rays)
        ] if self._num_rays > 1 else [heading]

        rays = [cast_ray(x, y, a) for a in angles]

        hit_dists = [math.hypot(ex - x, ey - y) for _, _, ex, ey, did_hit in rays if did_hit]
        min_dist = min(hit_dists) if hit_dists else _RAY_RANGE
        did_hit = bool(hit_dists)

        return [
            RangeReading(x, y, x + min_dist * math.cos(a), y + min_dist * math.sin(a), did_hit)
            for a in angles
        ]