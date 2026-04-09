from unittest.mock import patch

import numpy as np
import pytest

from robot_commander.depth_processing.cone_depth_processor import (
    ConeDepthProcessor,
    ConeGeometry,
)
from robot_commander.image_processing.intrinsics import Intrinsics


def _make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> Intrinsics:
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return Intrinsics(camera_matrix=camera_matrix, dist_coeffs=np.zeros(5), rms_error=0.0, image_size=(0, 0))


def _make_processor(
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 320.0,
    cy: float = 240.0,
    camera_T_sensor: np.ndarray | None = None,
    half_angle_degrees: float = 30.0,
) -> ConeDepthProcessor:
    if camera_T_sensor is None:
        camera_T_sensor = np.eye(4)
    intrinsics = _make_intrinsics(fx, fy, cx, cy)
    cone = ConeGeometry(half_angle_radians=np.radians(half_angle_degrees))
    with patch.object(ConeDepthProcessor, "__init__", lambda self, *args, **kwargs: None):
        processor = ConeDepthProcessor.__new__(ConeDepthProcessor)
        processor._depth_processor = None
        processor._intrinsics = intrinsics
        processor._camera_T_sensor = camera_T_sensor
        processor._cone_geometry = cone
    return processor


def test_cone_mask_principal_point_included():
    processor = _make_processor(cx=320.0, cy=240.0, half_angle_degrees=30.0)
    depth = np.ones((480, 640), dtype=np.float32)
    cone_mask = processor._compute_cone_mask(depth, processor._to_sensor_frame(depth))
    assert cone_mask[240, 320]


def test_cone_mask_far_corner_pixel_excluded():
    processor = _make_processor(fx=500.0, fy=500.0, cx=320.0, cy=240.0, half_angle_degrees=30.0)
    depth = np.ones((480, 640), dtype=np.float32)
    cone_mask = processor._compute_cone_mask(depth, processor._to_sensor_frame(depth))
    # Pixel (0, 0): X = (0 - 320) * 1 / 500 = -0.64, Y = (0 - 240) * 1 / 500 = -0.48
    # radial = sqrt(0.64^2 + 0.48^2) = 0.8, polar_angle = arctan2(0.8, 1) ≈ 38.7° > 30°
    assert not cone_mask[0, 0]


def test_cone_mask_points_behind_sensor_excluded():
    transform = np.diag([1.0, 1.0, -1.0, 1.0])
    processor = _make_processor(camera_T_sensor=transform, half_angle_degrees=90.0)
    depth = np.ones((480, 640), dtype=np.float32)
    cone_mask = processor._compute_cone_mask(depth, processor._to_sensor_frame(depth))
    assert not cone_mask.any()


def test_scale_calibration_identity_transform():
    processor = _make_processor(half_angle_degrees=90.0)
    flat_depth = np.full((480, 640), 2.0, dtype=np.float32)
    with patch.object(processor, "_get_raw_depth", return_value=flat_depth):
        result = processor.process(frame=np.zeros((480, 640, 3), dtype=np.uint8), ultrasonic_min_reading=4.0)
    np.testing.assert_allclose(result, 4.0, atol=1e-5)


def test_scale_with_translation_accounts_for_sensor_offset():
    transform = np.eye(4)
    transform[2, 3] = 0.5
    processor = _make_processor(cx=320.0, cy=240.0, camera_T_sensor=transform, half_angle_degrees=90.0)
    flat_depth = np.full((480, 640), 2.0, dtype=np.float32)
    # Principal point at depth 2.0 → sensor_z = 2.0 + 0.5 = 2.5
    # Ultrasonic = 2.5 → scale = 1.0 → output equals raw
    with patch.object(processor, "_get_raw_depth", return_value=flat_depth):
        result = processor.process(frame=np.zeros((480, 640, 3), dtype=np.uint8), ultrasonic_min_reading=2.5)
    np.testing.assert_allclose(result, flat_depth, atol=1e-5)


def test_scale_with_translation_requires_rescaling():
    transform = np.eye(4)
    transform[2, 3] = 0.5
    processor = _make_processor(cx=320.0, cy=240.0, camera_T_sensor=transform, half_angle_degrees=90.0)
    flat_depth = np.full((480, 640), 2.0, dtype=np.float32)
    # sensor_z = 2.5, ultrasonic = 5.0 → scale = 2.0
    with patch.object(processor, "_get_raw_depth", return_value=flat_depth):
        result = processor.process(frame=np.zeros((480, 640, 3), dtype=np.uint8), ultrasonic_min_reading=5.0)
    np.testing.assert_allclose(result, flat_depth * 2.0, atol=1e-5)

