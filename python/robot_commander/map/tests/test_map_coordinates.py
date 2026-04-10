import numpy as np
import pytest

from robot_commander.map.map_coordinates import MapCoordinates


def _make_map() -> MapCoordinates:
    return MapCoordinates(scale_px_per_m=100, width_px=400, height_px=400, origin_px=(200, 300))


def test_world_to_px_and_back_roundtrip():
    coords = _make_map()
    world_x, world_y = 0.5, 0.8
    px, py = coords.world_to_px(world_x, world_y)
    rx, ry = coords.px_to_world(px, py)
    assert abs(rx - world_x) < 0.01
    assert abs(ry - world_y) < 0.01


def test_to_map_px_y_axis_inverted():
    coords = _make_map()
    pts = np.array([[0.0, 1.0]])  # 1 m forward (positive y)
    px = coords.to_map_px(pts)
    ox, oy = coords.origin_px
    assert px[0, 0] == ox           # x unchanged
    assert px[0, 1] < oy            # positive world y maps upward in image


def test_camera_to_world_2d_raises_without_floor_basis():
    coords = _make_map()
    with pytest.raises(RuntimeError):
        coords.camera_to_world_2d(1.0, 0.0, 2.0)
