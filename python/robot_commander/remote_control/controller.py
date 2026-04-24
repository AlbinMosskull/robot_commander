import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from robot_commander import OccupancyMap, WorldPosition2d, plan_path_towards_goal
from robot_commander.config import load as load_config
from robot_commander.depth_processing.cone_depth_processor import ConeDepthProcessor
from robot_commander.depth_processing.cone_depth_rays import depth_to_rays
from robot_commander.depth_processing.depth_capture import DepthCapture, rays_to_ends
from robot_commander.sensor.range_reading import RangeReading
import robot_commander.depth_processing.depth_capture as depth_capture_io
from robot_commander.image_processing.intrinsics import Intrinsics
from robot_commander.localization.world_localizer import WorldLocalizer, WorldPose
from robot_commander.map.map_coordinates import MapCoordinates
from robot_commander.remote_control.agent_client import AgentClient

_PATH_COLLISION_MARGIN = 0.3
_OCC_RESOLUTION = 0.05
LOCALIZATION_LOST_THRESHOLD = 30
_ESCAPE_POSITION = (0.1, 0.2)
_FAILURES_DIR = Path(__file__).parent.parent / "debug_tools" / "failures"
_DEPTH_CAPTURE_PATH = Path(__file__).parent.parent / "debug_tools" / "latest_depth_capture.npz"
_LOGS_DIR = Path(__file__).parent.parent / "debug_tools" / "logs"
_DEPTH_RAY_RANGE_FACTOR = 1.5
_GAUSSIAN_SIGMA_M = 0.05


@dataclass
class MapState:
    agent_pos: WorldPose | None
    agent_heading: float | None
    planned_path: list[tuple[float, float]]
    checkpoint: tuple[float, float] | None
    occ_grid: np.ndarray = field(repr=False)


@dataclass
class PlanPathFailure:
    start: WorldPosition2d
    goal: WorldPosition2d
    collision_margin: float
    resolution: float
    origin_x: float
    origin_y: float
    occ_grid: np.ndarray = field(repr=False)


def _clip_ray(ray: RangeReading, max_length_m: float) -> tuple[float, float, float, float, bool]:
    dx = ray.end_x - ray.start_x
    dy = ray.end_y - ray.start_y
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length <= max_length_m:
        return ray.start_x, ray.start_y, ray.end_x, ray.end_y, ray.did_hit
    scale = max_length_m / length
    return ray.start_x, ray.start_y, ray.start_x + dx * scale, ray.start_y + dy * scale, False


class RemoteControl:
    def __init__(
        self,
        client: AgentClient | None,
        localizer: WorldLocalizer | None,
        cone_depth_processor: ConeDepthProcessor | None = None,
        cone_intrinsics: Intrinsics | None = None,
    ):
        self._client = client
        self._localizer = localizer
        self._cone_depth_processor = cone_depth_processor
        self._cone_intrinsics = cone_intrinsics

        self._map_coords = MapCoordinates.load(load_config().map.stencil_path)

        occ_origin_x = -self._map_coords.origin_px[0] / self._map_coords.scale_px_per_m
        occ_origin_y = (self._map_coords.origin_px[1] - self._map_coords.height_px) / self._map_coords.scale_px_per_m
        occ_width = round(self._map_coords.width_px / (self._map_coords.scale_px_per_m * _OCC_RESOLUTION))
        occ_height = round(self._map_coords.height_px / (self._map_coords.scale_px_per_m * _OCC_RESOLUTION))

        self._occ_resolution = _OCC_RESOLUTION
        self._occ_origin_x = occ_origin_x
        self._occ_origin_y = occ_origin_y
        self._occ_map = OccupancyMap(
            width=occ_width,
            height=occ_height,
            resolution=self._occ_resolution,
            origin_x=self._occ_origin_x,
            origin_y=self._occ_origin_y,
        )
        self._occ_lock = threading.Lock()

        self._checkpoint: tuple[float, float] | None = None
        self._planned_path: list[tuple[float, float]] = []
        self._agent_pos: WorldPose | None = None
        self._agent_heading: float | None = None
        self._localization_miss_count: int = LOCALIZATION_LOST_THRESHOLD
        self._last_escape_plan_time: float | None = None
        self._localization_jammed: bool = False
        self._pos_lock = threading.Lock()
        self._agent_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._agent_update_thread: threading.Thread | None = None
        self._agent_heading_thread: threading.Thread | None = None
        self._depth_worker_thread: threading.Thread | None = None
        self._depth_queue: queue.Queue = queue.Queue(maxsize=1)
        self._depth_captures_dir: Path = _LOGS_DIR / datetime.now().strftime("%Y%m%dT%H%M%S") / "depth_captures"
        self._depth_captures_dir.mkdir(parents=True, exist_ok=True)

    @property
    def map_coords(self) -> MapCoordinates:
        return self._map_coords

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._map_coords.width_px, self._map_coords.height_px

    def start(self) -> None:
        if self._client is None:
            return
        self._stop_event.clear()
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

    @property
    def localization_miss_count(self) -> int:
        with self._pos_lock:
            return self._localization_miss_count

    @property
    def connection_lost(self) -> bool:
        with self._pos_lock:
            return self._localization_miss_count >= LOCALIZATION_LOST_THRESHOLD

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
                self._localization_miss_count = 0
            else:
                self._localization_miss_count = min(
                    self._localization_miss_count + 1, LOCALIZATION_LOST_THRESHOLD
                )
        if pose is not None and self._client is not None:
            self._client.observe_position(pose.x, pose.y, pose.heading, confidence=1.0)
            escape_path = self._plan_path(WorldPosition2d(pose.x, pose.y), WorldPosition2d(*_ESCAPE_POSITION), "escape_plan.npz")
            if escape_path is not None:
                self._client.set_escape_plan(escape_path)
                with self._pos_lock:
                    self._last_escape_plan_time = time.monotonic()

    def snapshot(self) -> MapState:
        with self._pos_lock:
            agent_pos = self._agent_pos
            agent_heading = self._agent_heading
        with self._occ_lock:
            occ_grid = np.array(self._occ_map.get_grid(), dtype=np.float32)
        return MapState(
            agent_pos=agent_pos,
            agent_heading=agent_heading,
            planned_path=list(self._planned_path),
            checkpoint=self._checkpoint,
            occ_grid=occ_grid,
        )

    def set_offset_waypoint(self, angle_offset_rad: float, distance_m: float) -> None:
        with self._pos_lock:
            agent_pos = self._agent_pos
        if agent_pos is None or self._client is None:
            return
        target_x = agent_pos.x + distance_m * math.cos(agent_pos.heading + angle_offset_rad)
        target_y = agent_pos.y + distance_m * math.sin(agent_pos.heading + angle_offset_rad)
        self._checkpoint = (target_x, target_y)
        self._planned_path = []
        self._client.set_checkpoint(target_x, target_y)

    def handle_click(self, pixel_x: int, pixel_y: int, shift_held: bool) -> None:
        if self.connection_lost:
            return
        wx, wy = self._map_coords.px_to_world(pixel_x, pixel_y)
        if shift_held:
            with self._pos_lock:
                start = self._agent_pos
            if start is None or self._client is None:
                return
            result = self._plan_path(WorldPosition2d(start.x, start.y), WorldPosition2d(wx, wy), "user_path.npz")
            if result is None:
                return
            self._planned_path = result
            self._checkpoint = None
            self._client.set_path(self._planned_path)
        else:
            self._checkpoint = (wx, wy)
            self._planned_path = []
            if self._client is not None:
                self._client.set_checkpoint(wx, wy)

    def _plan_path(self, start: WorldPosition2d, goal: WorldPosition2d, failure_filename: str) -> list[tuple[float, float]] | None:
        with self._occ_lock:
            result = plan_path_towards_goal(self._occ_map, start, goal, _PATH_COLLISION_MARGIN)
            if result is None:
                failure = PlanPathFailure(
                    start=start,
                    goal=goal,
                    collision_margin=_PATH_COLLISION_MARGIN,
                    resolution=self._occ_resolution,
                    origin_x=self._occ_origin_x,
                    origin_y=self._occ_origin_y,
                    occ_grid=np.array(self._occ_map.get_grid(), dtype=np.float32),
                )
        if result is None:
            _FAILURES_DIR.mkdir(exist_ok=True)
            save_path = _FAILURES_DIR / failure_filename
            np.savez(
                save_path,
                occ_grid=failure.occ_grid,
                start=np.array([failure.start.x, failure.start.y]),
                goal=np.array([failure.goal.x, failure.goal.y]),
                collision_margin=np.array([failure.collision_margin]),
                resolution=np.array([failure.resolution]),
                origin=np.array([failure.origin_x, failure.origin_y]),
            )
            print(f"Path planning failed: {failure.start=} {failure.goal=} saved to {save_path}")
            return None
        return [(point.x, point.y) for point in result]

    @property
    def latest_agent_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._agent_frame

    def _depth_worker(self) -> None:
        while not self._stop_event.is_set():
            job = self._depth_queue.get()
            if job is None:
                break
            frame, ultrasonic_min, agent_pos, heading = job
            try:
                calibrated_depth, cone_mask = self._cone_depth_processor.process_with_mask(frame, ultrasonic_min)
                depth_rays = depth_to_rays(
                    calibrated_depth, self._cone_intrinsics,
                    agent_pos.x, agent_pos.y, heading,
                )
                capture = DepthCapture(
                    frame=frame,
                    calibrated_depth=calibrated_depth,
                    cone_mask=cone_mask,
                    ray_ends=rays_to_ends(depth_rays),
                    agent_x=agent_pos.x,
                    agent_y=agent_pos.y,
                    heading=heading,
                    ultrasonic_min=ultrasonic_min,
                    intrinsics=self._cone_intrinsics,
                )
                depth_capture_io.save(capture, _DEPTH_CAPTURE_PATH)
                depth_capture_io.save(capture, self._depth_captures_dir / f"{time.monotonic():.3f}.npz")
                max_ray_m = _DEPTH_RAY_RANGE_FACTOR * ultrasonic_min
                with self._occ_lock:
                    for ray in depth_rays:
                        try:
                            start_x, start_y, end_x, end_y, did_hit = _clip_ray(ray, max_ray_m)
                            if did_hit:
                                self._occ_map.ray_update_gaussian(start_x, start_y, end_x, end_y, _GAUSSIAN_SIGMA_M)
                            else:
                                self._occ_map.ray_update(start_x, start_y, end_x, end_y, False)
                        except Exception:
                            traceback.print_exc()
                    closest_rays = sorted(
                        depth_rays,
                        key=lambda r: (r.end_x - agent_pos.x) ** 2 + (r.end_y - agent_pos.y) ** 2,
                    )[:5]
                    print(f"occ map — {len(depth_rays)} rays | 5 closest endpoints:")
                    for ray in closest_rays:
                        dist = ((ray.end_x - agent_pos.x) ** 2 + (ray.end_y - agent_pos.y) ** 2) ** 0.5
                        value = self._occ_map.get_cell_value(ray.end_x, ray.end_y)
                        if not value:
                            print("Query for cell out of bounds")
                            continue
                        flag = " <-- CLEARED" if value is not None and value < 0.5 else ""
                        print(f"  ({ray.end_x:.2f}, {ray.end_y:.2f}) dist={dist:.2f}m cell={value:.3f}{flag}")
            except Exception:
                traceback.print_exc()

    def _stream_agent_heading(self) -> None:
        try:
            for _x, _y, heading in self._client.stream_positions():
                if self._stop_event.is_set():
                    break
                with self._pos_lock:
                    self._agent_heading = heading
        except Exception:
            traceback.print_exc()

    def _stream_agent_updates(self) -> None:
        try:
            for camera_frame_jpg, rays, cone in self._client.stream_agent_updates():
                if self._stop_event.is_set():
                    break
                if self.connection_lost:
                    continue
                if camera_frame_jpg is not None:
                    decoded = cv2.imdecode(np.frombuffer(camera_frame_jpg, np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None:
                        with self._frame_lock:
                            self._agent_frame = decoded
                with self._pos_lock:
                    agent_pos = self._agent_pos
                with self._occ_lock:
                    if rays:
                        for sx, sy, ex, ey, did_collide in rays:
                            try:
                                self._occ_map.ray_update(sx, sy, ex, ey, did_collide)
                            except Exception:
                                traceback.print_exc()
                if cone and self._cone_depth_processor is not None and self._cone_intrinsics is not None and agent_pos is not None:
                    ultrasonic_min, heading = cone
                    frame = self._agent_frame
                    try:
                        self._depth_queue.put_nowait((frame, ultrasonic_min, agent_pos, heading))
                    except queue.Full:
                        pass
        except Exception:
            traceback.print_exc()
