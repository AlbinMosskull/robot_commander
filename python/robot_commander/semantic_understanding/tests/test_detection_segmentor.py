import numpy as np

from robot_commander.semantic_understanding.detection_segmentor import _merge_boxes, _to_prompts
from robot_commander.semantic_understanding.types import BoundingBox, SegmentationResult


def test_merge_boxes_combines_overlapping():
    a = (0.9, BoundingBox(0, 0, 10, 10))
    b = (0.8, BoundingBox(5, 5, 15, 15))
    result = _merge_boxes([a, b])
    assert len(result) == 1
    assert result[0][1] == BoundingBox(0, 0, 15, 15)
    assert result[0][0] == 0.9  # max score


def test_merge_boxes_keeps_non_overlapping_separate():
    a = (0.9, BoundingBox(0, 0, 5, 5))
    b = (0.8, BoundingBox(10, 10, 15, 15))
    result = _merge_boxes([a, b])
    assert len(result) == 2


def test_merge_boxes_transitive():
    a = (0.9, BoundingBox(0, 0, 10, 10))
    b = (0.8, BoundingBox(8, 0, 18, 10))
    c = (0.7, BoundingBox(16, 0, 25, 10))
    result = _merge_boxes([a, b, c])
    assert len(result) == 1
    assert result[0][1] == BoundingBox(0, 0, 25, 10)


def test_to_prompts_merge_false_keeps_all():
    mask_a = np.zeros((20, 20), dtype=bool)
    mask_a[0:10, 0:10] = True
    mask_b = np.zeros((20, 20), dtype=bool)
    mask_b[5:15, 5:15] = True
    detections = [
        SegmentationResult(label="cat", score=0.9, mask=mask_a),
        SegmentationResult(label="cat", score=0.8, mask=mask_b),
    ]
    prompts = _to_prompts(detections, merge=False)
    assert len(prompts) == 2


def test_to_prompts_merge_true_merges_same_label():
    mask_a = np.zeros((20, 20), dtype=bool)
    mask_a[0:10, 0:10] = True
    mask_b = np.zeros((20, 20), dtype=bool)
    mask_b[5:15, 5:15] = True
    detections = [
        SegmentationResult(label="cat", score=0.9, mask=mask_a),
        SegmentationResult(label="cat", score=0.8, mask=mask_b),
    ]
    prompts = _to_prompts(detections, merge=True)
    assert len(prompts) == 1