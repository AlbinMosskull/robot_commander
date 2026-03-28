"""
Script to test localization
"""

import cv2
import numpy as np

from robot_commander.camera import intrinsics as calibration
from robot_commander.camera.camera import Camera
from robot_commander.camera.tag_detector import TagDetector, draw_tags
from robot_commander.config import load as load_config
from robot_commander.localization.localizer import Localizer

_cfg = load_config()

_TAG_SIZE = _cfg.tag.size_m


def main():
    cam_intrinsics = calibration.load()
    detector = TagDetector()
    localizer = Localizer(detector, cam_intrinsics.camera_matrix, _TAG_SIZE,
                          dist_coeffs=cam_intrinsics.dist_coeffs)

    with Camera(device_index=0, width=1920, height=1080) as cam:
        print("Camera opened. Press 'q' to quit.")
        cam.warm_up()

        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            tags = detector.detect(frame)
            annotated = draw_tags(frame, tags)

            pos = localizer.localize(frame)
            if pos is not None:
                x, y, z = pos
                dist = np.linalg.norm(pos)
                label = f"x={x:.3f}m  y={y:.3f}m  z={z:.3f}m, dist={dist:.3f}m"
                print(f"  {label}")
                cv2.putText(annotated, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(annotated, "No tag detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            display = cv2.resize(annotated, (_cfg.camera.preview_width, _cfg.camera.preview_height))
            cv2.imshow("Localization", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
