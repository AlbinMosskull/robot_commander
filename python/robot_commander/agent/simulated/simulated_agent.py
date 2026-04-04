import math
import threading
import time

from robot_commander.agent.abstract_agent import AbstractAgent
from robot_commander.agent.types import RangeReading
from robot_commander.agent.simulated.agent import Agent
from robot_commander.agent.simulated.ray_caster import cast_ray, _SWEEP_DEG, _SWEEP_DEG_PER_SEC

_TICK_HZ = 10


class SimulatedAgent(AbstractAgent):
    def __init__(self, start_x: float = 0.0, start_y: float = 0.0):
        self._agent = Agent(x=start_x, y=start_y, v=0.05)
        self._lock = threading.Lock()
        self._waypoints: list[tuple[float, float]] = []
        self._waypoint_idx: int = 0
        self._ray: RangeReading | None = None
        self._sweep_offset: float = -math.radians(_SWEEP_DEG / 2)
        self._sweep_dir: float = 1.0

        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def _tick_loop(self):
        while True:
            with self._lock:
                if self._waypoints:
                    wx, wy = self._waypoints[self._waypoint_idx]
                    self._agent.move(wx, wy)
                    if math.hypot(self._agent.x - wx, self._agent.y - wy) < 1e-3:
                        self._waypoint_idx += 1
                        if self._waypoint_idx >= len(self._waypoints):
                            self._waypoints = []

                x, y = self._agent.x, self._agent.y
                half_sweep = math.radians(_SWEEP_DEG / 2)
                step = math.radians(_SWEEP_DEG_PER_SEC / _TICK_HZ)
                self._sweep_offset += self._sweep_dir * step
                if self._sweep_offset >= half_sweep:
                    self._sweep_offset = half_sweep
                    self._sweep_dir = -1.0
                elif self._sweep_offset <= -half_sweep:
                    self._sweep_offset = -half_sweep
                    self._sweep_dir = 1.0
                sx, sy, ex, ey, did_hit = cast_ray(x, y, self._agent.heading + self._sweep_offset)
                self._ray = RangeReading(sx, sy, ex, ey, did_hit)

            time.sleep(1 / _TICK_HZ)

    def SetWaypointList(self, waypoints: list[tuple[float, float]]) -> None:
        with self._lock:
            self._waypoints = list(waypoints)
            self._waypoint_idx = 0

    def GetXandY(self) -> tuple[float, float]:
        with self._lock:
            return self._agent.x, self._agent.y

    def ObservePosition(self, x: float, y: float, confidence: float) -> None:
        pass  # Simulated agent has ground-truth position; no correction needed.

    def GetSensorReading(self) -> list[RangeReading]:
        with self._lock:
            if self._ray is None:
                return []
            return [self._ray]

    def GetCameraReading(self):
        return None
