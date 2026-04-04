import math
import threading
import time

from robot_commander.agent.abstract_agent import AbstractAgent
from robot_commander.agent.types import RangeReading
from robot_commander.agent.simulated.motion_model import (
    V_MAX_M_S,
    STEP_DURATION_S,
    WAYPOINT_THRESHOLD_M,
    advance_heading,
    advance_velocity,
    cardinal_direction,
)
from robot_commander.agent.simulated.sensors import SimulatedSensor, ConeSensor

_TICK_HZ = 10
_DT = 1.0 / _TICK_HZ


class SimulatedAgent(AbstractAgent):
    def __init__(self, start_x: float = 0.0, start_y: float = 0.0, sensor: SimulatedSensor | None = None):
        self.x = start_x
        self.y = start_y
        self.heading: float = math.pi / 2
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._step_timer: float = 0.0
        self._committed_vx: float = 0.0
        self._committed_vy: float = 0.0

        self._sensor = sensor if sensor is not None else ConeSensor()
        self._lock = threading.Lock()
        self._waypoints: list[tuple[float, float]] = []
        self._waypoint_idx: int = 0
        self._last_readings: list[RangeReading] = []

        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def _move(self, goal_x: float, goal_y: float) -> None:
        distance = math.hypot(goal_x - self.x, goal_y - self.y)

        if distance < WAYPOINT_THRESHOLD_M:
            self._vx, self._vy = advance_velocity(self._vx, self._vy, 0.0, 0.0, _DT)
            self.x += self._vx * _DT
            self.y += self._vy * _DT
            return

        self.heading = advance_heading(self.heading, goal_x, goal_y, self.x, self.y, _DT)

        self._step_timer -= _DT
        if self._step_timer <= 0:
            ux, uy = cardinal_direction(goal_x, goal_y, self.x, self.y, self.heading)
            self._vx, self._vy = advance_velocity(self._vx, self._vy, ux * V_MAX_M_S, uy * V_MAX_M_S, _DT)
            self._committed_vx = self._vx
            self._committed_vy = self._vy
            self._step_timer = STEP_DURATION_S
        else:
            self._vx = self._committed_vx
            self._vy = self._committed_vy

        self.x += self._vx * _DT
        self.y += self._vy * _DT

    def _tick_loop(self):
        while True:
            with self._lock:
                if self._waypoints:
                    wx, wy = self._waypoints[self._waypoint_idx]
                    self._move(wx, wy)
                    if math.hypot(self.x - wx, self.y - wy) < WAYPOINT_THRESHOLD_M:
                        self._waypoint_idx += 1
                        if self._waypoint_idx >= len(self._waypoints):
                            self._waypoints = []

                self._last_readings = self._sensor.read(self.x, self.y, self.heading)

            time.sleep(_DT)

    def SetWaypointList(self, waypoints: list[tuple[float, float]]) -> None:
        with self._lock:
            self._waypoints = list(waypoints)
            self._waypoint_idx = 0

    def GetXandY(self) -> tuple[float, float]:
        with self._lock:
            return self.x, self.y

    def ObservePosition(self, x: float, y: float, confidence: float) -> None:
        pass  # Simulated agent has ground-truth position; no correction needed.

    def GetSensorReading(self) -> list[RangeReading]:
        with self._lock:
            return list(self._last_readings)

    def GetCameraReading(self):
        return None
