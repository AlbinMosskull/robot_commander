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
    metric_model = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    processor = DepthProcessor(metric_model)
    print("Model loaded. Press 'q' to quit.")

    with Camera(device_index=0) as cam:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            depth = processor.process(frame)
            depth_vis = depth_to_colormap(depth)

            mid_y, mid_x = depth.shape[0] // 2, depth.shape[1] // 2
            mid_depth = depth[mid_y, mid_x]
            cv2.circle(depth_vis, (mid_x, mid_y), 5, (255, 255, 255), -1)
            cv2.putText(depth_vis, f"{mid_depth:.2f}", (mid_x + 10, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Camera", frame)
            cv2.imshow("Depth", depth_vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
