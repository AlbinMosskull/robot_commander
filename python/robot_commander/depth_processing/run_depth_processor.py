"""
Script to test depth processing on live camera frames.

Uses tag-calibrated Depth Anything V2 (non-metric model).
Show two AprilTags and press C to calibrate, then Q to quit.
"""

import cv2
import numpy as np

from robot_commander.image_processing import intrinsics as cal
from robot_commander.image_processing.camera import Camera
from robot_commander.image_processing.tag_detector import TagDetector, draw_tags
from robot_commander.config import load as load_config
from robot_commander.depth_processing.calibrated_depth_processor import CalibratedDepthProcessor
from robot_commander.localization.localizer import Localizer

_cfg = load_config()


def _depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    intrinsics = cal.load()
    detector = TagDetector()
    localizer = Localizer(detector, intrinsics.camera_matrix, _cfg.tag.size_m,
                          dist_coeffs=intrinsics.dist_coeffs)

    print("Loading Depth Anything V2 model...")
    processor = CalibratedDepthProcessor()
    print("Model loaded. Show 2 AprilTags and press C to calibrate. Press Q to quit.")

    with Camera() as cam:
        cam.warm_up()

        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            annotated = draw_tags(frame, detector.detect(frame))

            if processor.is_calibrated:
                depth = processor.process(frame)
                depth_vis = _depth_to_colormap(depth)

                mid_y, mid_x = depth.shape[0] // 2, depth.shape[1] // 2
                mid_depth = depth[mid_y, mid_x]
                cv2.circle(depth_vis, (mid_x, mid_y), 5, (255, 255, 255), -1)
                cv2.putText(depth_vis, f"{mid_depth:.3f} m", (mid_x + 10, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                display = cv2.resize(annotated, (640, 360))
                depth_display = cv2.resize(depth_vis, (640, 360))
                cv2.imshow("Depth", depth_display)
            else:
                n_tags = len(detector.detect(frame))
                status = f"{n_tags}/2 tags — press C to calibrate"
                cv2.putText(annotated, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                display = cv2.resize(annotated, (640, 360))

            cv2.imshow("Camera", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in (ord('c'), ord('C')):
                processor.calibrate(frame, localizer)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()