"""Live-feed UI for calibrating a CalibratedDepthProcessor."""

import cv2
import numpy as np

from robot_commander.image_processing.camera import Camera
from robot_commander.image_processing.tag_detector import TagDetector, draw_tags
from robot_commander.config import load as load_config
from robot_commander.depth_processing.calibrated_depth_processor import CalibratedDepthProcessor

_cfg = load_config()


def capture_calibration_frame(
    cam: Camera,
    processor: CalibratedDepthProcessor,
    detector: TagDetector,
) -> np.ndarray | None:
    """Show a live feed; user presses C when 2 AprilTags are visible to calibrate.

    Returns the captured frame on success, or None if the user presses Q.
    """
    window = "Show 2 AprilTags then press C to calibrate — Q to quit"
    while True:
        ok, frame = cam.read()
        if not ok:
            cv2.destroyWindow(window)
            return None

        tags = detector.detect(frame)
        vis = draw_tags(frame, tags)
        n = len(tags)
        color = (0, 255, 0) if n >= 2 else (0, 255, 255)
        cv2.putText(vis, f"{n}/2 tags visible — press C to calibrate",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow(window, cv2.resize(vis, (_cfg.camera.preview_width, _cfg.camera.preview_height)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window)
            return None
        if key in (ord('c'), ord('C')):
            if processor.calibrate(frame):
                cv2.destroyWindow(window)
                return frame
