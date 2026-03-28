import numpy as np

from robot_commander.image_processing.intrinsics import Intrinsics
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud


def _make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> Intrinsics:
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return Intrinsics(camera_matrix=K, dist_coeffs=np.zeros(5), rms_error=0.0, image_size=(0, 0))


def test_principal_point_pixel_projects_to_zero_xy():
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
    intrinsics = _make_intrinsics(fx, fy, cx, cy)

    depth = np.zeros((480, 640), dtype=np.float32)
    depth[int(cy), int(cx)] = 1.0  # one pixel at the principal point, depth = 1 m

    pts = depth_image_to_point_cloud(depth, intrinsics)

    assert len(pts) == 1
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 1.0], atol=1e-6)
