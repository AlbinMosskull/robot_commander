import math

import numpy as np
import pytest

from robot_commander.depth_processing.ultrasonic_plane_validator import (
    _MAX_NORMAL_ANGLE_DEG,
    _MIN_CONE_FILL_FRACTION,
    validate_ultrasonic_with_planes,
)


def _make_wall_points(n: int = 500, depth: float = 1.0, noise: float = 0.005) -> np.ndarray:
    rng = np.random.default_rng(42)
    xy = rng.uniform(-0.2, 0.2, (n, 2))
    z = np.full(n, depth) + rng.normal(0, noise, n)
    return np.stack([xy[:, 0], xy[:, 1], z], axis=-1)


def test_dominant_plane_selected_and_valid():
    points = _make_wall_points(n=500)
    result = validate_ultrasonic_with_planes(points)

    assert result.best_candidate is not None
    assert result.disqualification_reason is None
    assert result.best_candidate.cone_fill_fraction > _MIN_CONE_FILL_FRACTION
    assert result.best_candidate.normal_angle_deg < _MAX_NORMAL_ANGLE_DEG


def test_disqualified_by_low_cone_fill():
    rng = np.random.default_rng(0)
    many_random = rng.uniform(-0.5, 0.5, (1000, 3))
    many_random[:, 2] = np.abs(many_random[:, 2]) + 0.5

    result = validate_ultrasonic_with_planes(many_random, min_cone_fill_fraction=0.99)

    assert result.disqualification_reason is not None
    assert "fraction" in result.disqualification_reason


def test_disqualified_by_grazing_normal():
    rng = np.random.default_rng(1)
    n = 500
    xz = rng.uniform(-0.2, 0.2, (n, 2))
    y = np.full(n, 0.0) + rng.normal(0, 0.005, n)
    points = np.stack([xz[:, 0], y, xz[:, 1] + 1.0], axis=-1)

    result = validate_ultrasonic_with_planes(points, max_normal_angle_deg=10.0)

    assert result.disqualification_reason is not None
    assert "normal" in result.disqualification_reason


def test_empty_point_cloud_returns_no_planes():
    result = validate_ultrasonic_with_planes(np.zeros((0, 3)))

    assert result.best_candidate is None
    assert result.all_candidates == []
    assert result.disqualification_reason == "no planes detected"


def test_all_candidates_sorted_by_cone_fill_fraction():
    rng = np.random.default_rng(7)
    plane_a = _make_wall_points(n=400, depth=1.0)
    plane_b = _make_wall_points(n=50, depth=2.0)
    plane_b[:, 1] += 0.5
    points = np.vstack([plane_a, plane_b])

    result = validate_ultrasonic_with_planes(points)

    fractions = [c.cone_fill_fraction for c in result.all_candidates]
    assert fractions == sorted(fractions, reverse=True)
