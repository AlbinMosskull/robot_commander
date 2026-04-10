import numpy as np
import pytest

from robot_commander.map.map_geometry import _floor_basis, to_floor_2d, _filter_above_floor


def test_floor_basis_is_orthonormal_and_in_plane():
    normal = np.array([0.0, 1.0, 0.0])  # floor faces up along Y
    u, v = _floor_basis(normal)
    assert abs(np.linalg.norm(u) - 1.0) < 1e-6
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6
    assert abs(u @ v) < 1e-6          # orthogonal
    assert abs(u @ normal) < 1e-6     # u lies in the plane
    assert abs(v @ normal) < 1e-6     # v lies in the plane


def test_to_floor_2d_projects_along_basis():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0])
    points = np.array([[3.0, 99.0, 5.0]])  # y component should vanish
    result = to_floor_2d(points, u, v)
    assert np.allclose(result, [[3.0, 5.0]])


def test_filter_above_floor_keeps_only_valid_heights():
    normal = np.array([0.0, 1.0, 0.0])
    d_floor = -1.0  # floor at y = 1.0 (height = points @ normal - d = y + 1)
    points = np.array([
        [0.0, 0.90, 0.0],   # height 1.90 — too tall
        [0.0, 0.05, 0.0],   # height 1.05 — valid
        [0.0, -0.95, 0.0],  # height 0.05 — too short
    ])
    mask = _filter_above_floor(points, normal, d_floor)
    assert mask.tolist() == [False, True, False]
