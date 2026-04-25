import pytest

from robot_commander.remote_control.controller import _clip_ray, _halve_ray
from robot_commander.sensor.range_reading import RangeReading


def _ray(sx: float, sy: float, ex: float, ey: float, did_hit: bool = True) -> RangeReading:
    return RangeReading(start_x=sx, start_y=sy, end_x=ex, end_y=ey, did_hit=did_hit)


def test_halve_ray_returns_midpoint():
    ray = _ray(0.0, 0.0, 2.0, 0.0)
    sx, sy, ex, ey = _halve_ray(ray, max_length_m=10.0)
    assert sx == 0.0
    assert sy == 0.0
    assert ex == pytest.approx(1.0)
    assert ey == pytest.approx(0.0)


def test_halve_ray_clips_before_halving():
    ray = _ray(0.0, 0.0, 4.0, 0.0)
    sx, sy, ex, ey = _halve_ray(ray, max_length_m=2.0)
    assert ex == pytest.approx(1.0)
    assert ey == pytest.approx(0.0)


def test_halve_ray_diagonal():
    ray = _ray(0.0, 0.0, 2.0, 2.0)
    sx, sy, ex, ey = _halve_ray(ray, max_length_m=10.0)
    assert ex == pytest.approx(1.0)
    assert ey == pytest.approx(1.0)
