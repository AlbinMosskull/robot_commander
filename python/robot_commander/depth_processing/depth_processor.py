"""
Depth estimation using Depth Anything V2.
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


class DepthProcessor:
    """
    Runs Depth Anything V2 on camera frames to produce depth maps.

    Args:
        model: HuggingFace model ID. Defaults to the small V2 model for speed.
    """

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        self._pipe = pipeline(task="depth-estimation", model=model)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on a single BGR frame from OpenCV.

        Args:
            frame: BGR image as a numpy array (H, W, 3).

        Returns:
            Depth map as a float32 numpy array (H, W), with higher values
            indicating greater distance from the camera.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        result = self._pipe(pil_image)
        return np.array(result["depth"], dtype=np.float32)
