import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

import cv2
import numpy as np

from robot_commander import WorldPosition2d
from robot_commander.agent.adeept.adeept_transforms import CAMERA_T_SENSOR_CENTER
from robot_commander.config import load as load_config
from robot_commander.depth_processing.cone_depth_processor import ConeDepthProcessor, ConeGeometry
from robot_commander.depth_processing.depth_capture import DepthCapture
from robot_commander.depth_processing.depth_frame import DepthFrameInput
from robot_commander.image_processing import intrinsics as calibration
from robot_commander.image_processing.camera import Camera
from robot_commander.image_processing.intrinsics import AGENT_CAMERA_PATH, Intrinsics
from robot_commander.image_processing.tag_detector import TagDetector
from robot_commander.localization.camera_localizer import CameraLocalizer
from robot_commander.localization.localizer import Localizer
from robot_commander.localization.simulated_localizer import SimulatedLocalizer
from robot_commander.localization.world_localizer import WorldLocalizer, WorldPose
from robot_commander.map.map_coordinates import MapCoordinates
from robot_commander.remote_control.agent_client import AgentClient
from robot_commander.remote_control.navigation import Navigator
from robot_commander.remote_control.obstacle_mapping import ObstacleMapper

def build_controller(client: AgentClient | None, overhead_camera: Camera, simulated: bool = False) -> "RemoteControl":
    if client is None:
        return RemoteControl(None, None, overhead_camera=overhead_camera)

    if simulated:
        return RemoteControl(client, SimulatedLocalizer(client), overhead_camera=overhead_camera)

    cfg = load_config()
    overhead_intrinsics = calibration.load()
    agent_intrinsics = calibration.load(AGENT_CAMERA_PATH)

    detector = TagDetector()
    localizer = Localizer(detector, overhead_intrinsics.camera_matrix, cfg.tag.size_m,
                          dist_coeffs=overhead_intrinsics.dist_coeffs)
    map_coords = MapCoordinates.load(cfg.map.stencil_path)
    heading_offset = math.radians(cfg.agent.heading_offset_deg)
    camera_localizer = CameraLocalizer(localizer, map_coords, heading_offset=heading_offset)

    cone_geometry = ConeGeometry(half_angle_radians=math.radians(cfg.depth.cone_half_angle_deg))
    depth_processor = ConeDepthProcessor(
        intrinsics=agent_intrinsics,
        camera_T_sensor=CAMERA_T_SENSOR_CENTER,
        cone_geometry=cone_geometry,
    )

    return RemoteControl(client, camera_localizer, cone_depth_processor=depth_processor,
                         cone_intrinsics=agent_intrinsics, overhead_camera=overhead_camera)


LOCALIZATION_LOST_THRESHOLD_S = 1.0
_ESCAPE_POSITION = (0.2, 0.3)
_INITIAL_FREE_RADIUS_M = 0.3


@dataclass
class MapState:
    agent_pos: WorldPose | None
    agent_heading: float | None
    planned_path: list[tuple[float, float]]
    checkpoint: tuple[float, float] | None
    goal_heading: float | None
    occ_grid: np.ndarray = field(repr=False)
    escape_plan: list[tuple[float, float]] = field(default_factory=list)


class RemoteControl:
    def __init__(
        self,
        client: AgentClient | None,
        localizer: WorldLocalizer | None,
        cone_depth_processor: ConeDepthProcessor | None = None,
        cone_intrinsics: Intrinsics | None = None,
        overhead_camera: Camera | None = None,
    ):
        self._client = client
        self._localizer = localizer
        self._cone_depth_processor = cone_depth_processor
        self._overhead_camera = overhead_camera

        self._map_coords = MapCoordinates.load(load_config().map.stencil_path)

        self._obstacle_mapper = ObstacleMapper(
            self._map_coords, cone_depth_processor, cone_intrinsics
        )
        self._navigator = Navigator(
            client, self._obstacle_mapper, self._map_coords, self._get_agent_pos
        ) if client is not None else None

        self._agent_pos: WorldPose | None = None
        self._agent_heading: float | None = None
        self._localization_lost_since: float | None = time.monotonic()
        self._escape_plan: list[tuple[float, float]] = []
        self._last_escape_plan_time: float | None = None
        self._localization_jammed: bool = False
        self._robot_first_detected: bool = False
        self._pos_lock = threading.Lock()

        self._agent_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._overhead_frame: np.ndarray | None = None
        self._overhead_frame_lock = threading.Lock()
        self._payload_frame: np.ndarray | None = None
        self._payload_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._overhead_update_thread: threading.Thread | None = None
        self._agent_update_thread: threading.Thread | None = None
        self._agent_heading_thread: threading.Thread | None = None
        self._depth_worker_thread: threading.Thread | None = None
        self._depth_queue: queue.Queue = queue.Queue(maxsize=1)

    def _get_agent_pos(self) -> WorldPose | None:
        with self._pos_lock:
            return self._agent_pos

    @property
    def latest_depth_capture(self) -> DepthCapture | None:
        return self._obstacle_mapper.latest_depth_capture

    @property
    def map_coords(self) -> MapCoordinates:
        return self._map_coords

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._map_coords.width_px, self._map_coords.height_px

    def start(self) -> None:
        self._stop_event.clear()
        if self._overhead_camera is not None:
            self._overhead_update_thread = threading.Thread(target=self._stream_overhead_camera, daemon=True)
            self._overhead_update_thread.start()
        if self._client is None:
            return
        self._agent_update_thread = threading.Thread(target=self._stream_agent_updates, daemon=True)
        self._agent_update_thread.start()
        self._agent_heading_thread = threading.Thread(target=self._stream_agent_heading, daemon=True)
        self._agent_heading_thread.start()
        if self._cone_depth_processor is not None:
            self._depth_worker_thread = threading.Thread(target=self._depth_worker, daemon=True)
            self._depth_worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._depth_queue.put(None)
        if self._client is not None:
            self._client.close()
        if self._overhead_camera is not None:
            self._overhead_camera.release()

    @property
    def localization_lost_seconds(self) -> float | None:
        with self._pos_lock:
            if self._localization_lost_since is None:
                return None
            return time.monotonic() - self._localization_lost_since

    @property
    def connection_lost(self) -> bool:
        lost = self.localization_lost_seconds
        return lost is not None and lost >= LOCALIZATION_LOST_THRESHOLD_S

    @property
    def localization_jammed(self) -> bool:
        return self._localization_jammed

    def toggle_localization_jam(self) -> None:
        self._localization_jammed = not self._localization_jammed

    @property
    def escape_plan_age_s(self) -> float | None:
        with self._pos_lock:
            if self._last_escape_plan_time is None:
                return None
            return time.monotonic() - self._last_escape_plan_time

    def update(self, frame: np.ndarray) -> None:
        if self._localizer is None:
            return
        pose = None if self._localization_jammed else self._localizer.localize(frame)
        with self._pos_lock:
            if pose is not None:
                self._agent_pos = pose
                self._localization_lost_since = None
            elif self._localization_lost_since is None:
                self._localization_lost_since = time.monotonic()
        if pose is not None and not self._robot_first_detected:
            self._obstacle_mapper.mark_free_radius(pose.x, pose.y, _INITIAL_FREE_RADIUS_M)
            self._robot_first_detected = True
        if pose is not None and self._client is not None:
            self._client.observe_position(pose.x, pose.y, pose.heading, confidence=1.0)
            target = self._navigator.current_target() if self._navigator is not None else None
            escape_start = WorldPosition2d(*target) if target is not None else WorldPosition2d(pose.x, pose.y)
            escape_path = self._obstacle_mapper.plan_path(
                escape_start, WorldPosition2d(*_ESCAPE_POSITION), "escape_plan.npz"
            )
            if escape_path is not None:
                self._escape_plan = escape_path
                self._client.set_escape_plan(escape_path)
                with self._pos_lock:
                    self._last_escape_plan_time = time.monotonic()

    def snapshot(self) -> MapState:
        with self._pos_lock:
            agent_pos = self._agent_pos
            agent_heading = self._agent_heading
        occ_grid = self._obstacle_mapper.get_grid()
        if self._navigator is not None:
            planned_path, checkpoint, goal_heading = self._navigator.snapshot()
        else:
            planned_path, checkpoint, goal_heading = [], None, None
        return MapState(
            agent_pos=agent_pos,
            agent_heading=agent_heading,
            planned_path=planned_path,
            checkpoint=checkpoint,
            goal_heading=goal_heading,
            occ_grid=occ_grid,
            escape_plan=list(self._escape_plan),
        )

    def handle_click(self, pixel_x: int, pixel_y: int, shift_held: bool, goal_heading: float | None = None) -> None:
        if self.connection_lost or self._navigator is None:
            return
        self._navigator.handle_click(pixel_x, pixel_y, shift_held, goal_heading)

    def set_offset_waypoint(self, angle_offset_rad: float, distance_m: float) -> None:
        if self.connection_lost or self._navigator is None:
            return
        self._navigator.set_offset_waypoint(angle_offset_rad, distance_m)

    @property
    def latest_overhead_frame(self) -> np.ndarray | None:
        with self._overhead_frame_lock:
            return self._overhead_frame

    @property
    def latest_agent_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._agent_frame

    @property
    def latest_payload_frame(self) -> np.ndarray | None:
        with self._payload_lock:
            return self._payload_frame

    def enable_payload(self) -> None:
        if self._client is not None:
            self._client.enable_payload()

    def _depth_worker(self) -> None:
        while not self._stop_event.is_set():
            job = self._depth_queue.get()
            if job is None:
                break
            frame, ultrasonic_min, agent_pos, heading = job
            try:
                depth_input = DepthFrameInput(
                    frame=frame,
                    ultrasonic_min=ultrasonic_min,
                    agent_x=agent_pos.x,
                    agent_y=agent_pos.y,
                    agent_heading=heading,
                )
                self._obstacle_mapper.process_depth(depth_input)
            except Exception:
                traceback.print_exc()

    def _stream_overhead_camera(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self._overhead_camera.read()
            if ok:
                with self._overhead_frame_lock:
                    self._overhead_frame = frame
                self.update(frame)

    def _stream_agent_heading(self) -> None:
        try:
            for _, _, heading in self._client.stream_positions():
                if self._stop_event.is_set():
                    break
                with self._pos_lock:
                    self._agent_heading = heading
        except Exception:
            traceback.print_exc()

    def _stream_agent_updates(self) -> None:
        try:
            for camera_frame_jpg, rays, cone, payload_frame_jpg in self._client.stream_agent_updates():
                if self._stop_event.is_set():
                    break
                if not self.connection_lost:
                    if camera_frame_jpg is not None:
                        decoded = cv2.imdecode(np.frombuffer(camera_frame_jpg, np.uint8), cv2.IMREAD_COLOR)
                        if decoded is not None:
                            with self._frame_lock:
                                self._agent_frame = decoded
                    if payload_frame_jpg is not None:
                        decoded_payload = cv2.imdecode(np.frombuffer(payload_frame_jpg, np.uint8), cv2.IMREAD_COLOR)
                        if decoded_payload is not None:
                            with self._payload_lock:
                                self._payload_frame = decoded_payload
                    with self._pos_lock:
                        agent_pos = self._agent_pos
                    if rays:
                        self._obstacle_mapper.apply_rays(rays)
                    if cone and self._cone_depth_processor is not None and agent_pos is not None:
                        ultrasonic_min, heading = cone
                        frame = self._agent_frame
                        try:
                            self._depth_queue.put_nowait((frame, ultrasonic_min, agent_pos, heading))
                        except queue.Full:
                            pass
        except Exception:
            traceback.print_exc()
