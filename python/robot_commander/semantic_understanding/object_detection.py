import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from robot_commander.semantic_understanding.semantic_types import SegmentationResult


class ObjectDetector:
    """
    Runs object detection on camera frames and exposes results as rectangular masks.

    Uses the HuggingFace object-detection pipeline (default: DETR with ResNet-50).
    The bounding box for each detected object is converted to a boolean mask so the
    interface is identical to SemanticSegmentor.

    Args:
        model: HuggingFace model ID.
        threshold: Minimum confidence score to keep a detection.
    """

    def __init__(
        self,
        model: str = "facebook/detr-resnet-50",
        threshold: float = 0.5,
    ):
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            task="object-detection",
            model=model,
            device=device,
        )
        self._threshold = threshold

    def process(self, frame: np.ndarray) -> list[SegmentationResult]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        raw = self._pipe(pil_image, threshold=self._threshold)

        results = []
        for item in raw:
            box = item["box"]
            mask = np.zeros((h, w), dtype=bool)
            y1 = max(0, int(box["ymin"]))
            y2 = min(h, int(box["ymax"]))
            x1 = max(0, int(box["xmin"]))
            x2 = min(w, int(box["xmax"]))
            mask[y1:y2, x1:x2] = True
            results.append(
                SegmentationResult(
                    label=item["label"],
                    score=float(item["score"]),
                    mask=mask,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results
