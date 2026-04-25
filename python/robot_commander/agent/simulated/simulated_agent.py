import math
import threading
import time

import numpy as np

from robot_commander.agent.abstract_agent import AbstractAgent
from robot_commander.sensor.range_reading import RangeReading
from robot_commander.agent.simulated.motion_model import (
    V_MAX_M_S,
    WAYPOINT_THRESHOLD_M,
    HEADING_ALIGNMENT_RAD,
    advance_heading,
    advance_speed,
    normalize_angle,
)
from robot_commander.agent.simulated.sensors import SimulatedSensor, ConeSensor
from robot_commander.filtering.kalman_filter import KalmanFilter

_TICK_HZ = 10
_DT = 1.0 / _TICK_HZ
_REMOTE_TIMEOUT_S = 5.0

# Simulated GPS noise standard deviation (meters)
_GPS_NOISE_STD = 0.05


def _make_position_filter(start_x: float, start_y: float) -> KalmanFilter:
    I2 = np.eye(2)
    return KalmanFilter(
        F=I2,                               # Position doesn't self-evolve
        B=I2,                               # Control input is displacement [dx, dy]
        H=I2,                               # We measure position directly
        Q=np.eye(2) * 1e-4,                 # Small process noise
        R=np.eye(2) * _GPS_NOISE_STD**2,    # Measurement noise matches GPS std
        x0=np.array([start_x, start_y]),
        P0=I2,
    )


class SimulatedAgent(AbstractAgent):
    def __init__(self, start_x: float = 0.0, start_y: float = 0.0, sensor: SimulatedSensor | None = None):
        self.x = start_x
        self.y = start_y
        self.heading: float = math.pi / 2
        self._speed: float = 0.0

        self._position_filter = _make_position_filter(start_x, start_y)
        self._sensor = sensor if sensor is not None else ConeSensor()
        self._lock = threading.Lock()
        self._waypoints: list[tuple[float, float]] = []
        self._waypoint_idx: int = 0
        self._last_readings: list[RangeReading] = []

        self._last_remote_message_time: float = time.time()
        self._escape_plan: list[tuple[float, float]] = []
        self._escape_plan_idx: int = 0

        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def _notify_remote_message(self) -> None:
        self._last_remote_message_time = time.time()

    def _follow_waypoints(self, waypoints: list[tuple[float, float]], idx: int) -> tuple[list[tuple[float, float]], int]:
        if not waypoints:
            return waypoints, idx
        wx, wy = waypoints[idx]
        self._move(wx, wy)
        if math.hypot(self.x - wx, self.y - wy) < WAYPOINT_THRESHOLD_M:
            idx += 1
            if idx >= len(waypoints):
                return [], 0
        return waypoints, idx

    def _move(self, goal_x: float, goal_y: float) -> None:
        distance = math.hypot(goal_x - self.x, goal_y - self.y)

        if distance < WAYPOINT_THRESHOLD_M:
            self._speed = advance_speed(self._speed, 0.0, _DT)
            self.x += self._speed * math.cos(self.heading) * _DT
            self.y += self._speed * math.sin(self.heading) * _DT
            return

        self.heading = advance_heading(self.heading, goal_x, goal_y, self.x, self.y, _DT)

        heading_error = abs(normalize_angle(math.atan2(goal_y - self.y, goal_x - self.x) - self.heading))
        desired_speed = V_MAX_M_S if heading_error < HEADING_ALIGNMENT_RAD else 0.0
        self._speed = advance_speed(self._speed, desired_speed, _DT)

        self.x += self._speed * math.cos(self.heading) * _DT
        self.y += self._speed * math.sin(self.heading) * _DT

    def _tick_loop(self):
        while True:
            with self._lock:
                if time.time() - self._last_remote_message_time > _REMOTE_TIMEOUT_S and self._escape_plan:
                    self._escape_plan, self._escape_plan_idx = self._follow_waypoints(self._escape_plan, self._escape_plan_idx)
                else:
                    self._waypoints, self._waypoint_idx = self._follow_waypoints(self._waypoints, self._waypoint_idx)

                dx = self._speed * math.cos(self.heading) * _DT
                dy = self._speed * math.sin(self.heading) * _DT
                self._position_filter.predict(np.array([dx, dy]))

                noisy_gps = np.array([self.x, self.y]) + np.random.normal(0, _GPS_NOISE_STD, 2)
                self._position_filter.update(noisy_gps)

                self._last_readings = self._sensor.read(self.x, self.y, self.heading)

            time.sleep(_DT)

    def SetWaypointList(self, waypoints: list[tuple[float, float]], final_heading: float | None = None) -> None:
        with self._lock:
            self._waypoints = list(waypoints)
            self._waypoint_idx = 0
        self._notify_remote_message()

    def SetEscapePlan(self, waypoints: list[tuple[float, float]]) -> None:
        with self._lock:
            self._escape_plan = list(waypoints)
            self._escape_plan_idx = 0
        self._notify_remote_message()

    def GetXandY(self) -> tuple[float, float]:
        with self._lock:
            return float(self._position_filter.x[0]), float(self._position_filter.x[1])

    def ObservePosition(self, x: float, y: float, heading: float, confidence: float) -> None:
        self._notify_remote_message()  # Simulated agent has ground-truth position; no correction needed.

    def GetSensorReading(self) -> list[RangeReading]:
        with self._lock:
            return list(self._last_readings)

    def GetCameraReading(self):
        return None

    def GetHeading(self) -> float:
        with self._lock:
            return self.heading

    def GetUltrasonicMin(self) -> float | None:
        return None

    def RunCommand(self, command: str, duration_s: float) -> None:
        time.sleep(duration_s)

    def Scout(self) -> None:
        pass

    def EnablePayload(self) -> None:
        pass

    def GetPendingPayload(self) -> bytes | None:
        return None