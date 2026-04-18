from dataclasses import dataclass

import cv2
import numpy as np

from robot_commander.depth_processing.depth_processor import DepthProcessor
from robot_commander.image_processing.intrinsics import Intrinsics

_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"


@dataclass(frozen=True)
class ConeGeometry:
    half_angle_radians: float


class ConeDepthProcessor:
    def __init__(
        self,
        intrinsics: Intrinsics,
        camera_T_sensor: np.ndarray,
        cone_geometry: ConeGeometry,
        processing_width: int = 320,
    ) -> None:
        self._depth_processor = DepthProcessor(_MODEL)
        self._intrinsics = intrinsics
        self._camera_T_sensor = camera_T_sensor
        self._cone_geometry = cone_geometry
        self._processing_width = processing_width

    def process(self, frame: np.ndarray, ultrasonic_min_reading: float) -> np.ndarray:
        calibrated_depth, _ = self.process_with_mask(frame, ultrasonic_min_reading)
        return calibrated_depth

    def process_with_mask(
        self, frame: np.ndarray, ultrasonic_min_reading: float
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_depth = self._get_raw_depth(frame)
        sensor_points = self._to_sensor_frame(raw_depth)
        cone_mask = self._compute_cone_mask(raw_depth, sensor_points)
        if not cone_mask.any():
            raise ValueError("Cone mask is empty — no valid pixels in cone region")
        cone_min_sensor_depth = self._find_cone_minimum_in_sensor_frame(raw_depth, cone_mask, sensor_points)
        scale = ultrasonic_min_reading / cone_min_sensor_depth
        return (raw_depth * scale).astype(np.float32), cone_mask

    def _get_raw_depth(self, frame: np.ndarray) -> np.ndarray:
        original_h, original_w = frame.shape[:2]
        scale = self._processing_width / original_w
        small = cv2.resize(frame, (self._processing_width, int(original_h * scale)))
        small_depth = self._depth_processor.process(small)
        return cv2.resize(small_depth, (original_w, original_h))

    def _to_sensor_frame(self, depth: np.ndarray) -> np.ndarray:
        height, width = depth.shape
        column_grid, row_grid = np.meshgrid(
            np.arange(width, dtype=np.float64),
            np.arange(height, dtype=np.float64),
        )
        ones = np.ones((height, width), dtype=np.float64)
        camera_points = np.stack([
            (column_grid - self._intrinsics.cx) * depth / self._intrinsics.fx,
            (row_grid - self._intrinsics.cy) * depth / self._intrinsics.fy,
            depth.astype(np.float64),
            ones,
        ], axis=-1)
        return (camera_points @ self._camera_T_sensor.T)[..., :3]

    def _compute_cone_mask(self, depth: np.ndarray, sensor_points: np.ndarray) -> np.ndarray:
        sensor_z = sensor_points[..., 2]
        radial = np.sqrt(sensor_points[..., 0] ** 2 + sensor_points[..., 1] ** 2)
        polar_angle = np.arctan2(radial, sensor_z)
        return (depth > 0) & (sensor_z > 0) & (polar_angle < self._cone_geometry.half_angle_radians)

    def _find_cone_minimum_in_sensor_frame(
        self,
        depth: np.ndarray,
        cone_mask: np.ndarray,
        sensor_points: np.ndarray,
    ) -> float:
        masked_depth = np.where(cone_mask, depth, np.inf)
        row, col = np.unravel_index(np.argmin(masked_depth), depth.shape)
        return float(sensor_points[row, col, 2])
