import cv2
import numpy as np


def signal_lost_frame(width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 13, dtype=np.uint8)
    text = "SIGNAL LOST"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(canvas, text, ((width - tw) // 2, (height + th) // 2), font, scale, (60, 60, 180), thickness)
    return canvas
