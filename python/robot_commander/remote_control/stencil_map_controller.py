import threading
from pathlib import Path

import cv2
import numpy as np

from robot_commander import OccupancyMap, WorldPosition2d, plan_path
from robot_commander.map_building.map_coordinates import (
    _MAP_H,
    _MAP_W,
    _MAP_ORIGIN,
    _MAP_SCALE,
    px_to_world,
    world_to_px,
)
from robot_commander.remote_control.agent_client import AgentClient

_PATH_COLLISION_MARGIN = 0.08

_STENCIL_PATH = Path("plots/output/stencil_map.png")

_OCC_RESOLUTION = 0.05
_OCC_ORIGIN_X = -_MAP_ORIGIN[0] / _MAP_SCALE
_OCC_ORIGIN_Y = (_MAP_ORIGIN[1] - _MAP_H) / _MAP_SCALE
_OCC_WIDTH = round(_MAP_W / (_MAP_SCALE * _OCC_RESOLUTION))
_OCC_HEIGHT = round(_MAP_H / (_MAP_SCALE * _OCC_RESOLUTION))

_FREE_THRESHOLD = 0.3
_OCCUPIED_THRESHOLD = 0.7
_OVERLAY_ALPHA = 0.45


def _draw_occupancy_overlay(canvas: np.ndarray, occ_map: OccupancyMap) -> None:
    grid = np.array(occ_map.get_grid(), dtype=np.float32)
    grid = np.flipud(grid)
    grid = cv2.resize(grid, (_MAP_W, _MAP_H), interpolation=cv2.INTER_NEAREST)

    overlay = canvas.copy()
    overlay[grid < _FREE_THRESHOLD] = (0, 200, 0)
    overlay[grid > _OCCUPIED_THRESHOLD] = (0, 0, 200)

    known = (grid < _FREE_THRESHOLD) | (grid > _OCCUPIED_THRESHOLD)
    blended = cv2.addWeighted(overlay, _OVERLAY_ALPHA, canvas, 1 - _OVERLAY_ALPHA, 0)
    canvas[known] = blended[known]


class StencilMapController:
    def __init__(self, client: AgentClient | None):
        self._client = client

        if _STENCIL_PATH.exists():
            self._background = cv2.imread(str(_STENCIL_PATH))
        else:
            self._background = np.full((_MAP_H, _MAP_W, 3), 30, dtype=np.uint8)

        self._occ_map = OccupancyMap(
            width=_OCC_WIDTH,
            height=_OCC_HEIGHT,
            resolution=_OCC_RESOLUTION,
            origin_x=_OCC_ORIGIN_X,
            origin_y=_OCC_ORIGIN_Y,
        )
        self._occ_lock = threading.Lock()

        self._checkpoint: tuple[float, float] | None = None
        self._planned_path: list[tuple[float, float]] = []
        self._agent_pos: tuple[float, float] | None = None
        self._pos_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._pos_thread: threading.Thread | None = None
        self._rays_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._client is None:
            return
        self._stop_event.clear()
        self._pos_thread = threading.Thread(target=self._stream_pos, daemon=True)
        self._rays_thread = threading.Thread(target=self._stream_rays, daemon=True)
        self._pos_thread.start()
        self._rays_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._client is not None:
            self._client.close()
        if self._pos_thread is not None:
            self._pos_thread.join(timeout=2)
        if self._rays_thread is not None:
            self._rays_thread.join(timeout=2)

    def handle_click(self, pixel_x: int, pixel_y: int, shift_held: bool) -> None:
        wx, wy = px_to_world(pixel_x, pixel_y)
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

    def render(self) -> np.ndarray:
        canvas = self._background.copy()

        with self._occ_lock:
            _draw_occupancy_overlay(canvas, self._occ_map)

        if self._planned_path:
            pts = [world_to_px(wx, wy) for wx, wy in self._planned_path]
            for point_a, point_b in zip(pts, pts[1:]):
                cv2.line(canvas, point_a, point_b, (0, 200, 0), 2)
            cv2.circle(canvas, pts[-1], 8, (0, 200, 0), -1)
            cv2.circle(canvas, pts[-1], 8, (0, 0, 0), 1)
        elif self._checkpoint is not None:
            checkpoint_px = world_to_px(*self._checkpoint)
            cv2.circle(canvas, checkpoint_px, 8, (0, 200, 0), -1)
            cv2.circle(canvas, checkpoint_px, 8, (0, 0, 0), 1)

        with self._pos_lock:
            pos = self._agent_pos
        if pos is not None:
            agent_px = world_to_px(*pos)
            cv2.circle(canvas, agent_px, 8, (200, 80, 0), -1)
            cv2.circle(canvas, agent_px, 8, (0, 0, 0), 1)

        return canvas

    def _stream_pos(self) -> None:
        try:
            for x, y in self._client.stream_positions():
                if self._stop_event.is_set():
                    break
                with self._pos_lock:
                    self._agent_pos = (x, y)
                noisy_x, noisy_y = np.random.normal([x, y], scale=0.02)
                self._client.observe_position(float(noisy_x), float(noisy_y), confidence=1.0)
        except Exception:
            pass

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
                            pass
        except Exception:
            pass
