import math
import threading
import time

from robot_commander.agent.abstract_agent import AbstractAgent
from robot_commander.agent.types import RangeReading
from robot_commander.agent.simulated.agent import Agent
from robot_commander.agent.simulated.sensors import SimulatedSensor, ConeSensor

_TICK_HZ = 10


class SimulatedAgent(AbstractAgent):
    def __init__(self, start_x: float = 0.0, start_y: float = 0.0, sensor: SimulatedSensor | None = None):
        self._agent = Agent(x=start_x, y=start_y, v=0.05)
        self._sensor = sensor if sensor is not None else ConeSensor()
        self._lock = threading.Lock()
        self._waypoints: list[tuple[float, float]] = []
        self._waypoint_idx: int = 0
        self._last_readings: list[RangeReading] = []

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

                self._last_readings = self._sensor.read(
                    self._agent.x, self._agent.y, self._agent.heading
                )

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
            return list(self._last_readings)

    def GetCameraReading(self):
        return None
