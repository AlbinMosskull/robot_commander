import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robot_commander import OccupancyMap, WorldPosition2d, plan_path_towards_goal
from robot_commander.config import load as load_config
from robot_commander.localization.world_localizer import WorldLocalizer
from robot_commander.map_building.map_coordinates import MapCoordinates
from robot_commander.remote_control.agent_client import AgentClient

_PATH_COLLISION_MARGIN = 0.07
_OCC_RESOLUTION = 0.05
LOCALIZATION_LOST_THRESHOLD = 30
_ESCAPE_POSITION = (0.1, 0.2)
_FAILURES_DIR = Path(__file__).parent.parent / "debug_tools" / "failures"


@dataclass
class MapState:
    agent_pos: tuple[float, float] | None
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


class RemoteControl:
    def __init__(self, client: AgentClient | None, localizer: WorldLocalizer | None):
        self._client = client
        self._localizer = localizer

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
        self._agent_pos: tuple[float, float] | None = None
        self._localization_miss_count: int = LOCALIZATION_LOST_THRESHOLD
        self._last_escape_plan_time: float | None = None
        self._localization_jammed: bool = False
        self._pos_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._rays_thread: threading.Thread | None = None

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
        self._rays_thread = threading.Thread(target=self._stream_rays, daemon=True)
        self._rays_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._client is not None:
            self._client.close()
        if self._rays_thread is not None:
            self._rays_thread.join(timeout=2)

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
        pos = None if self._localization_jammed else self._localizer.localize(frame)
        with self._pos_lock:
            if pos is not None:
                self._agent_pos = pos
                self._localization_miss_count = 0
            else:
                self._localization_miss_count = min(
                    self._localization_miss_count + 1, LOCALIZATION_LOST_THRESHOLD
                )
        if pos is not None and self._client is not None:
            escape_path = self._plan_path(WorldPosition2d(pos[0], pos[1]), WorldPosition2d(*_ESCAPE_POSITION), "escape_plan.npz")
            if escape_path is not None:
                self._client.set_escape_plan(escape_path)
                with self._pos_lock:
                    self._last_escape_plan_time = time.monotonic()

    def snapshot(self) -> MapState:
        with self._pos_lock:
            agent_pos = self._agent_pos
        with self._occ_lock:
            occ_grid = np.array(self._occ_map.get_grid(), dtype=np.float32)
        return MapState(
            agent_pos=agent_pos,
            planned_path=list(self._planned_path),
            checkpoint=self._checkpoint,
            occ_grid=occ_grid,
        )

    def handle_click(self, pixel_x: int, pixel_y: int, shift_held: bool) -> None:
        if self.connection_lost:
            return
        wx, wy = self._map_coords.px_to_world(pixel_x, pixel_y)
        if shift_held:
            with self._pos_lock:
                start = self._agent_pos
            if start is None or self._client is None:
                return
            result = self._plan_path(WorldPosition2d(start[0], start[1]), WorldPosition2d(wx, wy), "user_path.npz")
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

    def _stream_rays(self) -> None:
        try:
            for rays in self._client.stream_rays():
                if self._stop_event.is_set():
                    break
                if self.connection_lost:
                    continue
                with self._occ_lock:
                    for sx, sy, ex, ey, did_collide in rays:
                        try:
                            self._occ_map.ray_update(sx, sy, ex, ey, did_collide)
                        except Exception:
                            traceback.print_exc()
        except Exception:
            traceback.print_exc()
