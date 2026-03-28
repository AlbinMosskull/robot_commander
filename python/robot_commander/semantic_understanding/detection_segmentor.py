import numpy as np

from robot_commander.semantic_understanding.object_detection import ObjectDetector
from robot_commander.semantic_understanding.sam_segmentor import SamSegmentor
from robot_commander.semantic_understanding.types import BoundingBox, SegmentationResult


def _merge_boxes(
    scored_boxes: list[tuple[float, BoundingBox]],
) -> list[tuple[float, BoundingBox]]:
    changed = True
    while changed:
        changed = False
        out: list[tuple[float, BoundingBox]] = []
        used = [False] * len(scored_boxes)
        for i, (sa, a) in enumerate(scored_boxes):
            if used[i]:
                continue
            for j in range(i + 1, len(scored_boxes)):
                if used[j]:
                    continue
                sb, b = scored_boxes[j]
                if a.overlaps(b):
                    a = a.merge(b)
                    sa = max(sa, sb)
                    used[j] = True
                    changed = True
            out.append((sa, a))
        scored_boxes = out
    return scored_boxes


def _to_prompts(
    detections: list[SegmentationResult],
    merge: bool,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    by_label: dict[str, list[tuple[float, BoundingBox]]] = {}
    for det in detections:
        box = BoundingBox.from_mask(det.mask)
        if box is None:
            continue
        by_label.setdefault(det.label, []).append((det.score, box))

    prompts: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for label, scored_boxes in by_label.items():
        if merge:
            scored_boxes = _merge_boxes(scored_boxes)
        for score, box in scored_boxes:
            prompts.append((label, score, box.as_tuple()))
    return prompts


class DetectionSegmentor:
    """
    Combines DETR object detection with SAM segmentation.

    Runs DETR to get bounding boxes, optionally merges overlapping boxes per
    label, then refines each with SAM to produce precise masks.

    Args:
        detector_model: HuggingFace model ID for object detection.
        sam_model: HuggingFace model ID for SAM.
        threshold: Minimum detection confidence score to keep.
        merge_boxes: If True, overlapping boxes for the same label are merged
            before being passed to SAM.
    """

    def __init__(
        self,
        detector_model: str = "facebook/detr-resnet-50",
        sam_model: str = "facebook/sam-vit-base",
        threshold: float = 0.5,
        merge_boxes: bool = True,
    ):
        self._detector = ObjectDetector(model=detector_model, threshold=threshold)
        self._sam = SamSegmentor(model=sam_model)
        self._merge_boxes = merge_boxes

    def process(self, frame: np.ndarray) -> list[SegmentationResult]:
        detections = self._detector.process(frame)
        prompts = _to_prompts(detections, merge=self._merge_boxes)
        return self._sam.process(frame, prompts)