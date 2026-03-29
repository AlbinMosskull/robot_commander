import threading
from pathlib import Path

import cv2
import numpy as np

from robot_commander import OccupancyMap, WorldPosition2d, plan_path
from robot_commander.map_building.map_coordinates import (
    _MAP_H, _MAP_W, _MAP_ORIGIN, _MAP_SCALE,
    px_to_world, world_to_px,
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


def main():
    if not _STENCIL_PATH.exists():
        print(f"Stencil map not found at {_STENCIL_PATH}. Run build_stencil_map.py first.")
        return

    background = cv2.imread(str(_STENCIL_PATH))
    client = AgentClient()
    occ_map = OccupancyMap(
        width=_OCC_WIDTH,
        height=_OCC_HEIGHT,
        resolution=_OCC_RESOLUTION,
        origin_x=_OCC_ORIGIN_X,
        origin_y=_OCC_ORIGIN_Y,
    )
    occ_lock = threading.Lock()

    checkpoint: tuple[float, float] | None = None
    planned_path: list[tuple[float, float]] = []
    agent_pos: tuple[float, float] | None = None
    pos_lock = threading.Lock()
    stop_event = threading.Event()

    def on_mouse(event, x, y, flags, param):
        nonlocal checkpoint, planned_path
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        wx, wy = px_to_world(x, y)
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            with pos_lock:
                start = agent_pos
            if start is None:
                return
            with occ_lock:
                result = plan_path(
                    occ_map,
                    WorldPosition2d(start[0], start[1]),
                    WorldPosition2d(wx, wy),
                    _PATH_COLLISION_MARGIN,
                )
            if result is None:
                print("No path found")
                return
            planned_path = [(p.x, p.y) for p in result]
            checkpoint = None
            client.set_path(planned_path)
        else:
            checkpoint = (wx, wy)
            planned_path = []
            client.set_checkpoint(wx, wy)

    def stream_pos_thread():
        nonlocal agent_pos
        try:
            for x, y in client.stream_positions():
                if stop_event.is_set():
                    break
                with pos_lock:
                    agent_pos = (x, y)
        except Exception:
            pass

    def stream_rays_thread():
        try:
            for rays in client.stream_rays():
                if stop_event.is_set():
                    break
                with occ_lock:
                    for sx, sy, ex, ey, did_collide in rays:
                        try:
                            occ_map.ray_update(sx, sy, ex, ey, did_collide)
                        except Exception:
                            pass
        except Exception:
            pass

    pos_thread = threading.Thread(target=stream_pos_thread)
    rays_thread = threading.Thread(target=stream_rays_thread)
    pos_thread.start()
    rays_thread.start()

    cv2.namedWindow("Agent Map")
    cv2.setMouseCallback("Agent Map", on_mouse)

    print("Left-click: set checkpoint. Shift+click: plan path. Press 'q' to quit.")
    try:
        while True:
            canvas = background.copy()

            with occ_lock:
                _draw_occupancy_overlay(canvas, occ_map)

            if planned_path:
                pts = [world_to_px(wx, wy) for wx, wy in planned_path]
                for a, b in zip(pts, pts[1:]):
                    cv2.line(canvas, a, b, (0, 200, 0), 2)
                cv2.circle(canvas, pts[-1], 8, (0, 200, 0), -1)
                cv2.circle(canvas, pts[-1], 8, (0, 0, 0), 1)
            elif checkpoint is not None:
                px = world_to_px(*checkpoint)
                cv2.circle(canvas, px, 8, (0, 200, 0), -1)
                cv2.circle(canvas, px, 8, (0, 0, 0), 1)

            with pos_lock:
                pos = agent_pos
            if pos is not None:
                px = world_to_px(*pos)
                cv2.circle(canvas, px, 8, (200, 80, 0), -1)
                cv2.circle(canvas, px, 8, (0, 0, 0), 1)

            cv2.imshow("Agent Map", canvas)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        client.close()
        pos_thread.join(timeout=2)
        rays_thread.join(timeout=2)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
