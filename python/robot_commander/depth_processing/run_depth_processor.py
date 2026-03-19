"""
Script to test depth processing on live camera frames.
"""

import cv2
import numpy as np

from robot_commander.camera.camera import Camera
from robot_commander.depth_processing.depth_processor import DepthProcessor


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    gray = normalized.astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def main():
    print("Loading Depth Anything V2 model...")
    processor = DepthProcessor()
    print("Model loaded. Press 'q' to quit.")

    with Camera(device_index=0) as cam:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            depth = processor.process(frame)
            depth_vis = depth_to_colormap(depth)

            cv2.imshow("Camera", frame)
            cv2.imshow("Depth", depth_vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
