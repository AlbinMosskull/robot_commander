import math
import threading
import time

import cv2
import numpy as np
from picamera2 import Picamera2

from robot_commander.agent.abstract_agent import AbstractAgent
from robot_commander.sensor.range_reading import RangeReading
from robot_commander.filtering.kalman_filter import KalmanFilter
from robot_commander.agent.adeept.hardware.Move import RaspClaws
from robot_commander.agent.adeept.hardware import Ultra

_ULTRA_HIT_THRESHOLD_CM = 190.0
_TICK_HZ = 10
_DT = 1.0 / _TICK_HZ
_REMOTE_TIMEOUT_S = 5.0
_WAYPOINT_THRESHOLD_M = 0.15
_CAMERA_WIDTH = 640
_CAMERA_HEIGHT = 480


def _make_position_filter() -> KalmanFilter:
    identity = np.eye(2)
    return KalmanFilter(
        F=identity,
        B=identity,
        H=identity,
        Q=np.eye(2) * 1e-4,
        R=np.eye(2) * 0.1,
        x0=np.zeros(2),
        P0=identity,
    )


def _direction_command(current_x: float, current_y: float, target_x: float, target_y: float, heading: float) -> str:
    target_angle = math.atan2(target_y - current_y, target_x - current_x)
    relative_angle = (target_angle - heading + math.pi) % (2 * math.pi) - math.pi
    if abs(relative_angle) < math.pi / 4:
        return "forward"
    if abs(relative_angle) > 3 * math.pi / 4:
        return "backward"
    return "left" if relative_angle > 0 else "right"


class AdeeptAgent(AbstractAgent):
    def __init__(self):
        self._robot = RaspClaws()
        self._robot.daemon = True
        self._robot.start()

        self._camera = Picamera2()
        camera_config = self._camera.create_preview_configuration(
            main={"size": (_CAMERA_WIDTH, _CAMERA_HEIGHT), "format": "RGB888"}
        )
        self._camera.configure(camera_config)
        self._camera.start()

        self._position_filter = _make_position_filter()
        self._lock = threading.Lock()
        self._waypoints: list[tuple[float, float]] = []
        self._waypoint_idx: int = 0
        self._escape_plan: list[tuple[float, float]] = []
        self._escape_plan_idx: int = 0
        self._last_remote_message_time: float = time.time()

        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def _follow_waypoints(
        self,
        waypoints: list[tuple[float, float]],
        index: int,
    ) -> tuple[list[tuple[float, float]], int]:
        if not waypoints:
            self._robot.command_input("stand")
            return waypoints, index

        current_x, current_y = float(self._position_filter.x[0]), float(self._position_filter.x[1])
        target_x, target_y = waypoints[index]

        if math.hypot(current_x - target_x, current_y - target_y) < _WAYPOINT_THRESHOLD_M:
            index += 1
            if index >= len(waypoints):
                self._robot.command_input("stand")
                return [], 0
            target_x, target_y = waypoints[index]

        command = _direction_command(current_x, current_y, target_x, target_y, heading=0.0)
        self._robot.command_input(command)
        return waypoints, index

    def _tick_loop(self) -> None:
        while True:
            with self._lock:
                remote_timed_out = time.time() - self._last_remote_message_time > _REMOTE_TIMEOUT_S
                if remote_timed_out and self._escape_plan:
                    self._escape_plan, self._escape_plan_idx = self._follow_waypoints(
                        self._escape_plan, self._escape_plan_idx
                    )
                else:
                    self._waypoints, self._waypoint_idx = self._follow_waypoints(
                        self._waypoints, self._waypoint_idx
                    )
                self._position_filter.predict(np.zeros(2))
            time.sleep(_DT)

    def SetWaypointList(self, waypoints: list[tuple[float, float]]) -> None:
        with self._lock:
            self._waypoints = list(waypoints)
            self._waypoint_idx = 0
            self._last_remote_message_time = time.time()

    def SetEscapePlan(self, waypoints: list[tuple[float, float]]) -> None:
        with self._lock:
            self._escape_plan = list(waypoints)
            self._escape_plan_idx = 0
            self._last_remote_message_time = time.time()

    def GetXandY(self) -> tuple[float, float]:
        with self._lock:
            return float(self._position_filter.x[0]), float(self._position_filter.x[1])

    def ObservePosition(self, x: float, y: float, heading: float, confidence: float) -> None:
        with self._lock:
            self._position_filter.update(np.array([x, y]))
            self._last_remote_message_time = time.time()

    def GetSensorReading(self) -> list[RangeReading]:
        return []

    def GetCameraReading(self) -> np.ndarray | None:
        frame = self._camera.capture_array()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def GetHeading(self) -> float:
        return 0.0

    def GetUltrasonicMin(self) -> float | None:
        distance_cm = Ultra.checkdist()
        if distance_cm >= _ULTRA_HIT_THRESHOLD_CM:
            return None
        return distance_cm / 100.0
