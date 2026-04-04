import threading
import traceback
from dataclasses import dataclass, field

import numpy as np

from robot_commander import OccupancyMap, WorldPosition2d, plan_path
from robot_commander.config import load as load_config
from robot_commander.localization.world_localizer import WorldLocalizer
from robot_commander.map_building.map_coordinates import MapCoordinates
from robot_commander.remote_control.agent_client import AgentClient

_PATH_COLLISION_MARGIN = 0.08
_OCC_RESOLUTION = 0.05
LOCALIZATION_LOST_THRESHOLD = 30


@dataclass
class MapState:
    agent_pos: tuple[float, float] | None
    planned_path: list[tuple[float, float]]
    checkpoint: tuple[float, float] | None
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

        self._occ_map = OccupancyMap(
            width=occ_width,
            height=occ_height,
            resolution=_OCC_RESOLUTION,
            origin_x=occ_origin_x,
            origin_y=occ_origin_y,
        )
        self._occ_lock = threading.Lock()

        self._checkpoint: tuple[float, float] | None = None
        self._planned_path: list[tuple[float, float]] = []
        self._agent_pos: tuple[float, float] | None = None
        self._localization_miss_count: int = LOCALIZATION_LOST_THRESHOLD
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

    def update(self, frame: np.ndarray) -> None:
        if self._localizer is None:
            return
        pos = self._localizer.localize(frame)
        with self._pos_lock:
            if pos is not None:
                self._agent_pos = pos
                self._localization_miss_count = 0
            else:
                self._localization_miss_count = min(
                    self._localization_miss_count + 1, LOCALIZATION_LOST_THRESHOLD
                )

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
        wx, wy = self._map_coords.px_to_world(pixel_x, pixel_y)
        if shift_held:
            with self._pos_lock:
                start = self._agent_pos
            if start is None or self._client is None:
                return
            with self._occ_lock:
                result = plan_path(
                    self._occ_map,
                    WorldPosition2d(start[0], start[1]),
                    WorldPosition2d(wx, wy),
                    _PATH_COLLISION_MARGIN,
                )
            if result is None:
                return
            self._planned_path = [(point.x, point.y) for point in result]
            self._checkpoint = None
            self._client.set_path(self._planned_path)
        else:
            self._checkpoint = (wx, wy)
            self._planned_path = []
            if self._client is not None:
                self._client.set_checkpoint(wx, wy)

    def _stream_rays(self) -> None:
        try:
            for rays in self._client.stream_rays():
                if self._stop_event.is_set():
                    break
                with self._occ_lock:
                    for sx, sy, ex, ey, did_collide in rays:
                        try:
                            self._occ_map.ray_update(sx, sy, ex, ey, did_collide)
                        except Exception:
                            traceback.print_exc()
        except Exception:
            traceback.print_exc()
