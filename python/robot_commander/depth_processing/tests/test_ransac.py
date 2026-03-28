import numpy as np

from robot_commander.depth_processing.ransac import detect_planes


def test_detects_axis_aligned_plane():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, (200, 3)).astype(np.float32)
    points[:, 1] = 0.0

    planes = detect_planes(points, n_planes=1, n_iterations=100, distance_threshold=0.01, seed=0)

    assert len(planes) == 1
    assert planes[0].inliers.sum() == len(points)
    np.testing.assert_allclose(np.abs(planes[0].normal), [0.0, 1.0, 0.0], atol=1e-6)
