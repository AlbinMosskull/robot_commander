import math
import threading
import time

import numpy as np
from picamera2 import Picamera2

from robot_commander.agent.abstract_agent import AbstractAgent
from robot_commander.sensor.range_reading import RangeReading
from robot_commander.filtering.kalman_filter import KalmanFilter
from robot_commander.filtering.heading_filter import HeadingFilter
from robot_commander.agent.adeept import adeept_transforms
from robot_commander.agent.adeept.hardware.Move import RaspClaws, _DEPTH_SENSOR_PAN_CHANNEL
from robot_commander.agent.adeept.hardware import Ultra
from robot_commander.agent.adeept.hardware.mpu6050_gyro import Mpu6050Gyro
from robot_commander.agent.adeept.run_logger import RunLogger
from robot_commander.agent.adeept.adeept_motion_model import (
    WAYPOINT_THRESHOLD_M,
    normalize_angle,
    predict_displacement,
)

_ULTRA_HIT_THRESHOLD_CM = 190.0
_STAND_SETTLE_S = 2.0
_SCOUT_OFFSETS_RAD = [math.radians(d) for d in (-30, 30)]
_SCOUT_DWELL_S = 7.0
_SCOUT_HEADING_THRESHOLD_RAD = math.radians(5)
_SWEEP_RANGE_DEG = 45
_SWEEP_STEP_DEG = 5
_SWEEP_STEP_INTERVAL_S = 0.01
_SWEEP_SETTLE_S = 0.05  # one full echo cycle at max range (2m round-trip ≈ 12ms)
_HEADING_ENTRY_RAD = math.radians(15)
_HEADING_EXIT_RAD = math.radians(30)
_TICK_HZ = 10
_DT = 1.0 / _TICK_HZ
_REMOTE_TIMEOUT_S = 5.0
_CAMERA_WIDTH = 1296
_CAMERA_HEIGHT = 972

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
    def __init__(self, escape_plan_enabled: bool = True, raw_sensor: bool = False):
        self._robot = RaspClaws()
        self._robot.daemon = True
        self._robot.start()

        self._camera = Picamera2()
        camera_config = self._camera.create_preview_configuration(
            main={"size": (_CAMERA_WIDTH, _CAMERA_HEIGHT), "format": "RGB888"}
        )
        self._camera.configure(camera_config)
        self._camera.start()

        self._gyro = Mpu6050Gyro()
        self._position_filter = _make_position_filter()
        self._heading_filter = HeadingFilter(
            initial_heading=0.0,
            process_noise=0.01,
            measurement_noise=0.1,
        )
        self._current_command: str = "stand"
        self._heading_aligned: bool = False
        self._manual_override: bool = False
        self._lock = threading.Lock()
        self._waypoints: list[tuple[float, float]] = []
        self._waypoint_idx: int = 0
        self._escape_plan: list[tuple[float, float]] = []
        self._escape_plan_idx: int = 0
        self._last_remote_message_time: float = time.time()

        self._escape_plan_enabled = escape_plan_enabled
        self._gyro_heading: float | None = None
        self._last_motion_time: float = 0.0
        self._logger = RunLogger()

        self._raw_sensor = raw_sensor
        self._stop_event = threading.Event()
        if raw_sensor:
            self._sweep_rays: list[RangeReading] = []
            self._sweep_lock = threading.Lock()
            threading.Thread(target=self._run_sweep, daemon=True).start()

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
            self._heading_aligned = False
            if index >= len(waypoints):
                self._current_command = "stand"
                self._robot.command_input("stand")
                return [], 0
            target_x, target_y = waypoints[index]

        heading_error = normalize_angle(math.atan2(target_y - current_y, target_x - current_x) - self._heading_filter.heading)

        if abs(heading_error) < _HEADING_ENTRY_RAD:
            self._heading_aligned = True
        elif abs(heading_error) > _HEADING_EXIT_RAD:
            self._heading_aligned = False

        if self._heading_aligned:
            command = "forward"
        else:
            command = "left" if heading_error > 0 else "right"

        self._current_command = command
        self._last_motion_time = time.time()
        self._robot.command_input(command)
        return waypoints, index

    def _tick_loop(self) -> None:
        while not self._stop_event.is_set():
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
                gyro_delta = self._gyro.z_angular_velocity_rad_s() * _DT
                self._heading_filter.predict(gyro_delta)
                if self._gyro_heading is not None:
                    self._gyro_heading = normalize_angle(self._gyro_heading + gyro_delta)
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
                log_gyro_heading = self._gyro_heading
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
                gyro_heading=log_gyro_heading,
            )
            time.sleep(_DT)

    def SetWaypointList(self, waypoints: list[tuple[float, float]]) -> None:
        with self._lock:
            self._waypoints = list(waypoints)
            self._waypoint_idx = 0
            self._heading_aligned = False
            self._last_remote_message_time = time.time()

    def SetEscapePlan(self, waypoints: list[tuple[float, float]]) -> None:
        if not self._escape_plan_enabled:
            return
        with self._lock:
            self._escape_plan = list(waypoints)
            self._escape_plan_idx = 0
            self._heading_aligned = False
            self._last_remote_message_time = time.time()

    def GetXandY(self) -> tuple[float, float]:
        with self._lock:
            return float(self._position_filter.x[0]), float(self._position_filter.x[1])

    def ObservePosition(self, x: float, y: float, heading: float, confidence: float) -> None:
        with self._lock:
            pos_innovation = self._position_filter.update(np.array([x, y]))
            heading_innovation = self._heading_filter.update(heading)
            self._last_remote_message_time = time.time()
            if self._gyro_heading is None:
                self._gyro_heading = heading
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

    def _run_sweep(self) -> None:
        try:
            self._run_sweep_loop()
        except Exception:
            import traceback
            traceback.print_exc()

    def _run_sweep_loop(self) -> None:
        center = float(self._robot.init_angles[_DEPTH_SENSOR_PAN_CHANNEL])
        angle_deg = center
        direction = 1
        while not self._stop_event.is_set():
            self._robot.set_servo_angle(_DEPTH_SENSOR_PAN_CHANNEL, angle_deg)
            time.sleep(_SWEEP_SETTLE_S)
            print(f"sweep {angle_deg:.0f}°", flush=True)
            distance_cm = Ultra.checkdist()
            if distance_cm < _ULTRA_HIT_THRESHOLD_CM:
                ray = self._servo_reading_to_ray(angle_deg, center, distance_cm / 100.0)
                with self._sweep_lock:
                    self._sweep_rays.append(ray)
            angle_deg += direction * _SWEEP_STEP_DEG
            if angle_deg >= center + _SWEEP_RANGE_DEG:
                angle_deg = center + _SWEEP_RANGE_DEG
                direction = -1
            elif angle_deg <= center - _SWEEP_RANGE_DEG:
                angle_deg = center - _SWEEP_RANGE_DEG
                direction = 1
            time.sleep(_SWEEP_STEP_INTERVAL_S)

    def _servo_reading_to_ray(
        self, servo_angle_deg: float, center_angle_deg: float, distance_m: float
    ) -> RangeReading:
        offset_rad = math.radians(servo_angle_deg - center_angle_deg)
        robot_T_s = adeept_transforms.robot_T_sensor(offset_rad)
        sensor_origin = robot_T_s[:3, 3]
        sensor_forward = robot_T_s[:3, 2]
        hit_robot = sensor_origin + distance_m * sensor_forward
        heading = self.GetHeading()
        agent_x, agent_y = self.GetXandY()
        c, s = math.cos(heading), math.sin(heading)
        start_x = agent_x + s * sensor_origin[0] + c * sensor_origin[2]
        start_y = agent_y - c * sensor_origin[0] + s * sensor_origin[2]
        hit_world_x = agent_x + s * hit_robot[0] + c * hit_robot[2]
        hit_world_y = agent_y - c * hit_robot[0] + s * hit_robot[2]
        return RangeReading(start_x, start_y, hit_world_x, hit_world_y, did_hit=True)

    def GetSensorReading(self) -> list[RangeReading]:
        if not self._raw_sensor:
            return []
        with self._sweep_lock:
            rays = list(self._sweep_rays)
            self._sweep_rays.clear()
        return rays

    def GetCameraReading(self) -> np.ndarray | None:
        return self._camera.capture_array()

    def GetHeading(self) -> float:
        with self._lock:
            return self._heading_filter.heading

    def GetUltrasonicMin(self) -> float | None:
        if self._raw_sensor:
            return None
        with self._lock:
            if self._current_command != "stand":
                return None
            if time.time() - self._last_motion_time < _STAND_SETTLE_S:
                return None
        distance_cm = Ultra.checkdist()
        if distance_cm >= _ULTRA_HIT_THRESHOLD_CM:
            return None
        return distance_cm / 100.0

    def close(self) -> None:
        self._stop_event.set()
        if self._raw_sensor:
            center = float(self._robot.init_angles[_DEPTH_SENSOR_PAN_CHANNEL])
            self._robot.set_servo_angle(_DEPTH_SENSOR_PAN_CHANNEL, center)
            self._robot.release_servo(_DEPTH_SENSOR_PAN_CHANNEL)
        self._robot.command_input("stand")
        self._robot.cleanup()
        self._camera.stop()
        self._camera.close()
        Ultra.sensor.close()
        self._logger.close()

    def _rotate_to_heading(self, target_rad: float) -> None:
        while True:
            with self._lock:
                error = normalize_angle(target_rad - self._heading_filter.heading)
            if abs(error) < _SCOUT_HEADING_THRESHOLD_RAD:
                break
            command = "left" if error > 0 else "right"
            with self._lock:
                self._current_command = command
                self._last_motion_time = time.time()
            self._robot.command_input(command)
            time.sleep(_DT)
        with self._lock:
            self._current_command = "stand"
        self._robot.command_input("stand")

    def Scout(self) -> None:
        with self._lock:
            self._manual_override = True
            original_heading = self._heading_filter.heading
        try:
            for offset_rad in _SCOUT_OFFSETS_RAD:
                self._rotate_to_heading(normalize_angle(original_heading + offset_rad))
                time.sleep(_SCOUT_DWELL_S)
            self._rotate_to_heading(original_heading)
        finally:
            with self._lock:
                self._manual_override = False

    def RunCommand(self, command: str, duration_s: float) -> None:
        with self._lock:
            self._manual_override = True
            self._last_motion_time = time.time()
        try:
            self._robot.command_input(command)
            time.sleep(duration_s)
            self._robot.command_input("stand")
        finally:
            with self._lock:
                self._manual_override = False
