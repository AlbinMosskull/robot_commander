import math
import threading
import time

import cv2
import numpy as np
from picamera2 import Picamera2

from robot_commander.agent.abstract_agent import AbstractAgent
from robot_commander.sensor.range_reading import RangeReading
from robot_commander.filtering.kalman_filter import KalmanFilter
from robot_commander.filtering.heading_filter import HeadingFilter
from robot_commander.agent.adeept.hardware.Move import RaspClaws
from robot_commander.agent.adeept.hardware import Ultra
from robot_commander.agent.adeept.run_logger import RunLogger
from robot_commander.agent.adeept.adeept_motion_model import (
    OMEGA_MAX_RAD_S,
    WAYPOINT_THRESHOLD_M,
    normalize_angle,
    direction_command,
    predict_displacement,
)

_ULTRA_HIT_THRESHOLD_CM = 190.0
_TICK_HZ = 10
_DT = 1.0 / _TICK_HZ
_REMOTE_TIMEOUT_S = 5.0
_CAMERA_WIDTH = 1920
_CAMERA_HEIGHT = 1080
_IDLE_SERVO_CHANNELS = [12, 13, 14, 15]


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


class AdeeptAgent(AbstractAgent):
    def __init__(self):
        self._robot = RaspClaws()
        self._robot.daemon = True
        self._robot.start()
        for ch in _IDLE_SERVO_CHANNELS:
            self._robot.release_servo(ch)

        self._camera = Picamera2()
        camera_config = self._camera.create_preview_configuration(
            main={"size": (_CAMERA_WIDTH, _CAMERA_HEIGHT), "format": "RGB888"}
        )
        self._camera.configure(camera_config)
        self._camera.start()

        self._position_filter = _make_position_filter()
        self._heading_filter = HeadingFilter(
            initial_heading=0.0,
            process_noise=0.01,
            measurement_noise=0.1,
        )
        self._current_command: str = "stand"
        self._manual_override: bool = False
        self._lock = threading.Lock()
        self._waypoints: list[tuple[float, float]] = []
        self._waypoint_idx: int = 0
        self._escape_plan: list[tuple[float, float]] = []
        self._escape_plan_idx: int = 0
        self._last_remote_message_time: float = time.time()

        self._logger = RunLogger()

        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def _follow_waypoints(
        self,
        waypoints: list[tuple[float, float]],
        index: int,
    ) -> tuple[list[tuple[float, float]], int]:
        if self._manual_override:
            return waypoints, index
        if not waypoints:
            self._current_command = "stand"
            self._robot.command_input("stand")
            return waypoints, index

        current_x, current_y = float(self._position_filter.x[0]), float(self._position_filter.x[1])
        target_x, target_y = waypoints[index]

        if math.hypot(current_x - target_x, current_y - target_y) < WAYPOINT_THRESHOLD_M:
            index += 1
            if index >= len(waypoints):
                self._current_command = "stand"
                self._robot.command_input("stand")
                return [], 0
            target_x, target_y = waypoints[index]

        heading_error = normalize_angle(math.atan2(target_y - current_y, target_x - current_x) - self._heading_filter.heading)
        command = direction_command(heading_error)
        self._current_command = command
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
                    active_waypoints = self._escape_plan
                    active_idx = self._escape_plan_idx
                else:
                    self._waypoints, self._waypoint_idx = self._follow_waypoints(
                        self._waypoints, self._waypoint_idx
                    )
                    active_waypoints = self._waypoints
                    active_idx = self._waypoint_idx
                current_heading = self._heading_filter.heading
                current_x = float(self._position_filter.x[0])
                current_y = float(self._position_filter.x[1])
                dx, dy = predict_displacement(self._current_command, current_heading, _DT)
                self._position_filter.predict(np.array([dx, dy]))
                if self._current_command == "left":
                    self._heading_filter.predict(OMEGA_MAX_RAD_S * _DT)
                elif self._current_command == "right":
                    self._heading_filter.predict(-OMEGA_MAX_RAD_S * _DT)
                else:
                    self._heading_filter.predict(0.0)
                if active_waypoints and active_idx < len(active_waypoints):
                    target_x, target_y = active_waypoints[active_idx]
                    heading_error_log = normalize_angle(
                        math.atan2(target_y - current_y, target_x - current_x) - current_heading
                    )
                    distance_log = math.hypot(current_x - target_x, current_y - target_y)
                else:
                    target_x = target_y = heading_error_log = distance_log = None
                log_heading = self._heading_filter.heading
                log_variance = self._heading_filter.variance
                log_command = self._current_command
            self._logger.log_tick(
                heading=log_heading,
                heading_variance=log_variance,
                pos_x=current_x,
                pos_y=current_y,
                command=log_command,
                heading_error=heading_error_log,
                distance_to_target=distance_log,
                target_x=target_x,
                target_y=target_y,
            )
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
            pos_innovation = self._position_filter.update(np.array([x, y]))
            heading_innovation = self._heading_filter.update(heading)
            self._last_remote_message_time = time.time()
        self._logger.log_observation(
            observed_heading=heading,
            heading_innovation=heading_innovation,
            observed_x=x,
            observed_y=y,
            pos_innovation_x=float(pos_innovation[0]),
            pos_innovation_y=float(pos_innovation[1]),
        )
        if heading_innovation is None:
            print(
                f"filter innovation — position: ({pos_innovation[0]:+.3f}, {pos_innovation[1]:+.3f}) m  "
                f"heading: rejected (outlier)"
            )
        else:
            print(
                f"filter innovation — position: ({pos_innovation[0]:+.3f}, {pos_innovation[1]:+.3f}) m  "
                f"heading: {math.degrees(heading_innovation):+.1f}°"
            )

    def GetSensorReading(self) -> list[RangeReading]:
        return []

    def GetCameraReading(self) -> np.ndarray | None:
        frame = self._camera.capture_array()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def GetHeading(self) -> float:
        with self._lock:
            return self._heading_filter.heading

    def GetUltrasonicMin(self) -> float | None:
        distance_cm = Ultra.checkdist()
        if distance_cm >= _ULTRA_HIT_THRESHOLD_CM:
            return None
        return distance_cm / 100.0

    def RunCommand(self, command: str, duration_s: float) -> None:
        with self._lock:
            self._manual_override = True
        try:
            self._robot.command_input(command)
            time.sleep(duration_s)
            self._robot.command_input("stand")
        finally:
            with self._lock:
                self._manual_override = False
