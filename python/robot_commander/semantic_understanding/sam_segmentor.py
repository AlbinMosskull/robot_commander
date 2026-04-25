import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from robot_commander.semantic_understanding.semantic_types import SegmentationResult


class SamSegmentor:
    def __init__(self, model: str = "facebook/sam-vit-base"):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = SamModel.from_pretrained(model).to(self._device)
        self._processor = SamProcessor.from_pretrained(model)

    def process(
        self,
        frame: np.ndarray,
        prompts: list[tuple[str, float, tuple[int, int, int, int]]],
    ) -> list[SegmentationResult]:
        """
        Run SAM on a frame with bounding-box prompts.

        Args:
            frame: BGR numpy array (H, W, 3).
            prompts: list of (label, score, (x1, y1, x2, y2)).

        Returns:
            One SegmentationResult per prompt with the SAM-refined mask.
        """
        if not prompts:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        results = []
        for label, score, (x1, y1, x2, y2) in prompts:
            inputs = self._processor(
                images=pil_image,
                input_boxes=[[[x1, y1, x2, y2]]],
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            # masks is a list (one per image); masks[0] shape: (1, num_masks, H, W)
            mask_tensor = masks[0][0]  # (num_masks, H, W)
            iou_scores = outputs.iou_scores[0][0]  # (num_masks,)
            best_idx = int(iou_scores.argmax())
            mask = mask_tensor[best_idx].numpy().astype(bool)

            results.append(SegmentationResult(label=label, score=score, mask=mask))

        return results
