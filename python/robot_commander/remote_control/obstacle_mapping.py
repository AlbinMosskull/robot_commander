import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from robot_commander import OccupancyMap, WorldPosition2d, plan_path_towards_goal_theta_star
from robot_commander.depth_processing.cone_depth_processor import ConeDepthProcessor
from robot_commander.depth_processing.depth_capture import DepthCapture, rays_to_ends
from robot_commander.depth_processing.depth_frame import (
    DepthFrameInput,
    DepthFrameResult,
    process_depth_frame,
)
import robot_commander.depth_processing.depth_capture as depth_capture_io
from robot_commander.image_processing.intrinsics import Intrinsics
from robot_commander.map.map_coordinates import MapCoordinates

_OCC_RESOLUTION = 0.05
_GAUSSIAN_SIGMA_M = 0.01
_PATH_COLLISION_MARGIN = 0.1
_FAILURES_DIR = Path(__file__).parent.parent / "debug_tools" / "failures"
_DEPTH_CAPTURE_PATH = Path(__file__).parent.parent / "debug_tools" / "latest_depth_capture.npz"
_LOGS_DIR = Path(__file__).parent.parent / "debug_tools" / "logs"


@dataclass
class PlanPathFailure:
    start: WorldPosition2d
    goal: WorldPosition2d
    collision_margin: float
    resolution: float
    origin_x: float
    origin_y: float
    occ_grid: np.ndarray = field(repr=False)


class ObstacleMapper:
    def __init__(
        self,
        map_coords: MapCoordinates,
        cone_depth_processor: ConeDepthProcessor | None = None,
        cone_intrinsics: Intrinsics | None = None,
    ):
        self._processor = cone_depth_processor
        self._intrinsics = cone_intrinsics

        occ_origin_x = -map_coords.origin_px[0] / map_coords.scale_px_per_m
        occ_origin_y = (map_coords.origin_px[1] - map_coords.height_px) / map_coords.scale_px_per_m
        occ_width = round(map_coords.width_px / (map_coords.scale_px_per_m * _OCC_RESOLUTION))
        occ_height = round(map_coords.height_px / (map_coords.scale_px_per_m * _OCC_RESOLUTION))

        self._occ_resolution = _OCC_RESOLUTION
        self._occ_origin_x = occ_origin_x
        self._occ_origin_y = occ_origin_y
        self._occ_map = OccupancyMap(
            width=occ_width,
            height=occ_height,
            resolution=self._occ_resolution,
            origin_x=occ_origin_x,
            origin_y=occ_origin_y,
        )
        self._occ_lock = threading.Lock()

        self._latest_depth_capture: DepthCapture | None = None
        self._depth_capture_lock = threading.Lock()
        self._depth_captures_dir: Path = (
            _LOGS_DIR / datetime.now().strftime("%Y%m%dT%H%M%S") / "depth_captures"
        )
        self._depth_captures_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_depth_capture(self) -> DepthCapture | None:
        with self._depth_capture_lock:
            return self._latest_depth_capture

    def mark_free_radius(self, x: float, y: float, radius_m: float) -> None:
        with self._occ_lock:
            self._occ_map.mark_free_radius(x, y, radius_m)

    def get_grid(self) -> np.ndarray:
        with self._occ_lock:
            return np.array(self._occ_map.get_grid(), dtype=np.float32)

    def plan_path(
        self,
        start: WorldPosition2d,
        goal: WorldPosition2d,
        failure_filename: str,
    ) -> list[tuple[float, float]] | None:
        with self._occ_lock:
            result = plan_path_towards_goal_theta_star(
                self._occ_map, start, goal, _PATH_COLLISION_MARGIN
            )
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

    def apply_rays(self, rays: list[tuple[float, float, float, float, bool]]) -> None:
        with self._occ_lock:
            for sx, sy, ex, ey, did_collide in rays:
                try:
                    self._occ_map.ray_update(sx, sy, ex, ey, did_collide)
                except Exception:
                    traceback.print_exc()

    def process_depth(self, depth_input: DepthFrameInput) -> DepthFrameResult:
        result = process_depth_frame(depth_input, self._processor, self._intrinsics)
        self._apply_occupancy_rays(result)
        capture = self._make_depth_capture(result)
        depth_capture_io.save(capture, _DEPTH_CAPTURE_PATH)
        depth_capture_io.save(capture, self._depth_captures_dir / f"{time.monotonic():.3f}.npz")
        with self._depth_capture_lock:
            self._latest_depth_capture = capture
        return result

    def _apply_occupancy_rays(self, result: DepthFrameResult) -> None:
        with self._occ_lock:
            for sx, sy, ex, ey in result.occupancy_rays.gaussian_hit_rays:
                try:
                    self._occ_map.ray_update_gaussian(sx, sy, ex, ey, _GAUSSIAN_SIGMA_M)
                except Exception:
                    traceback.print_exc()
            for sx, sy, ex, ey in result.occupancy_rays.free_rays:
                try:
                    self._occ_map.ray_update(sx, sy, ex, ey, False)
                except Exception:
                    traceback.print_exc()

    def _make_depth_capture(self, result: DepthFrameResult) -> DepthCapture:
        if result.is_calibrated:
            depth = result.calibrated_depth
            ray_ends = rays_to_ends(result.depth_rays)
        else:
            depth = result.raw_depth
            ray_ends = np.empty((0, 2), dtype=np.float32)
        return DepthCapture(
            frame=result.frame,
            depth=depth,
            cone_mask=result.cone_mask,
            ray_ends=ray_ends,
            agent_x=result.agent_x,
            agent_y=result.agent_y,
            heading=result.agent_heading,
            ultrasonic_min=result.ultrasonic_min,
            intrinsics=result.intrinsics,
            is_calibrated=result.is_calibrated,
        )
