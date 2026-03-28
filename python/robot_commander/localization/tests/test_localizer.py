from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robot_commander.image_processing.tag_detector import DetectedTag
from robot_commander.localization.localizer import Localizer

_BLANK_FRAME = np.zeros((300, 300, 3), dtype=np.uint8)
_CAMERA_MATRIX = np.array([[500, 0, 150], [0, 500, 150], [0, 0, 1]], dtype=np.float64)
_TAG_SIZE = 0.1  # metres


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


def _make_localizer(detector):
    return Localizer(detector, _CAMERA_MATRIX, _TAG_SIZE)


def test_localize_returns_none_when_no_tags_visible():
    localizer = _make_localizer(_mock_detector([]))
    assert localizer.localize(_BLANK_FRAME) is None


def test_localize_returns_none_when_solvepnp_fails():
    tag = _make_tag(0, 150, 150)
    localizer = _make_localizer(_mock_detector([tag]))
    with patch("cv2.solvePnP", return_value=(False, None, None)):
        assert localizer.localize(_BLANK_FRAME) is None


def test_localize_returns_metric_position():
    tag = _make_tag(0, 150, 150)
    localizer = _make_localizer(_mock_detector([tag]))
    tvec = np.array([[0.1], [-0.05], [1.2]])
    with patch("cv2.solvePnP", return_value=(True, np.zeros((3, 1)), tvec)):
        result = localizer.localize(_BLANK_FRAME)
    assert result == pytest.approx((0.1, -0.05, 1.2))


def test_localize_uses_first_detected_tag():
    tag_a = _make_tag(0, 100, 100)
    tag_b = _make_tag(1, 200, 200)
    localizer = _make_localizer(_mock_detector([tag_a, tag_b]))
    tvec = np.array([[0.0], [0.0], [1.0]])
    with patch("cv2.solvePnP", return_value=(True, np.zeros((3, 1)), tvec)) as mock_pnp:
        localizer.localize(_BLANK_FRAME)
    np.testing.assert_array_equal(mock_pnp.call_args_list[0][0][1], tag_a.corners)
