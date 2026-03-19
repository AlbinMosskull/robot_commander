from unittest.mock import MagicMock

import numpy as np
import pytest

from robot_commander.camera.tag_detector import DetectedTag
from robot_commander.localization.localizer import Localizer

_BLANK_FRAME = np.zeros((300, 300, 3), dtype=np.uint8)


def _make_tag(tag_id: int, cx: float, cy: float) -> DetectedTag:
    corners = np.array(
        [[cx - 10, cy - 10], [cx + 10, cy - 10], [cx + 10, cy + 10], [cx - 10, cy + 10]],
        dtype=np.float32,
    )
    return DetectedTag(tag_id=tag_id, corners=corners, center=(cx, cy))


def _mock_detector(*call_results):
    detector = MagicMock()
    detector.detect.side_effect = list(call_results)
    return detector


def test_init_raises_when_no_tags_in_init_frame():
    detector = _mock_detector([])
    with pytest.raises(RuntimeError):
        Localizer(detector, _BLANK_FRAME)


def test_init_selects_lowest_y_as_origin():
    tag_top = _make_tag(0, 100, 50)
    tag_bottom = _make_tag(1, 100, 200)
    detector = _mock_detector([tag_top, tag_bottom])
    localizer = Localizer(detector, _BLANK_FRAME)
    assert localizer._origin.tag_id == 0


def test_localize_returns_none_when_only_one_tag_visible():
    origin = _make_tag(0, 100, 100)
    detector = _mock_detector([origin], [origin])
    localizer = Localizer(detector, _BLANK_FRAME)
    assert localizer.localize(_BLANK_FRAME) is None


def test_localize_returns_none_when_no_tags_visible():
    origin = _make_tag(0, 100, 100)
    detector = _mock_detector([origin], [])
    localizer = Localizer(detector, _BLANK_FRAME)
    assert localizer.localize(_BLANK_FRAME) is None


def test_localize_returns_correct_offset():
    origin = _make_tag(0, 100, 100)
    target = _make_tag(1, 250, 180)
    detector = _mock_detector([origin, target], [origin, target])
    localizer = Localizer(detector, _BLANK_FRAME)
    assert localizer.localize(_BLANK_FRAME) == (150.0, 80.0)


def test_localize_returns_negative_offset_when_target_is_above_and_left():
    origin = _make_tag(0, 200, 50)
    target = _make_tag(1, 50, 200)
    # In localize frame, target moves to upper-left of origin
    target_moved = _make_tag(1, 30, 20)
    detector = _mock_detector([origin, target], [origin, target_moved])
    localizer = Localizer(detector, _BLANK_FRAME)
    assert localizer.localize(_BLANK_FRAME) == (30 - 200, 20 - 50)


def test_localize_identifies_origin_by_proximity_when_it_moves():
    origin_init = _make_tag(0, 100, 100)
    origin_moved = _make_tag(0, 108, 104)
    target = _make_tag(1, 250, 180)
    detector = _mock_detector([origin_init, target], [origin_moved, target])
    localizer = Localizer(detector, _BLANK_FRAME)
    result = localizer.localize(_BLANK_FRAME)
    assert result == (250 - 108, 180 - 104)
