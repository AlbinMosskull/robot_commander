import numpy as np

from robot_commander.semantic_understanding.semantic_types import BoundingBox


def test_overlapping_boxes_detected():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 15, 15)
    assert a.overlaps(b)


def test_non_overlapping_boxes_not_detected():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(11, 0, 20, 10)
    assert not a.overlaps(b)


def test_merge_produces_union_bounds():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 20, 20)
    assert a.merge(b) == BoundingBox(0, 0, 20, 20)


def test_from_mask_returns_tight_box():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 5:15] = True
    box = BoundingBox.from_mask(mask)
    assert box == BoundingBox(x1=5, y1=10, x2=14, y2=19)


def test_from_mask_empty_returns_none():
    assert BoundingBox.from_mask(np.zeros((50, 50), dtype=bool)) is None