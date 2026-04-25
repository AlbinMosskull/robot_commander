import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from robot_commander.semantic_understanding.semantic_types import SegmentationResult


class SemanticSegmentor:
    """
    Runs Mask2Former instance segmentation on camera frames.

    Args:
        model: HuggingFace model ID.
        threshold: Minimum confidence score to keep a detection.
    """

    def __init__(
        self,
        model: str = "facebook/mask2former-swin-large-coco-instance",
        threshold: float = 0.5,
    ):
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            task="image-segmentation",
            model=model,
            device=device,
        )
        self._threshold = threshold

    def process(self, frame: np.ndarray) -> list[SegmentationResult]:
        """
        Run instance segmentation on a single BGR frame from OpenCV.

        Args:
            frame: BGR image as a numpy array (H, W, 3).

        Returns:
            List of SegmentationResult, one per detected instance, sorted by
            score descending.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        raw = self._pipe(pil_image, threshold=self._threshold)

        results = []
        for item in raw:
            mask = np.array(item["mask"], dtype=bool)
            results.append(
                SegmentationResult(
                    label=item["label"],
                    score=float(item["score"]),
                    mask=mask,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results
