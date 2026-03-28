"""
Script to open the webcam and visualize detected ArUco tags.
"""

import cv2

from robot_commander.image_processing.camera import WebCamera
from robot_commander.image_processing.tag_detector import TagDetector, draw_tags
from robot_commander.config import load as load_config

_cfg = load_config()


def main():
    detector = TagDetector()

    with WebCamera() as cam:
        print("Camera opened. Press 'q' to quit.")
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            tags = detector.detect(frame)
            annotated = draw_tags(frame, tags)

            for tag in tags:
                print(f"  Tag ID={tag.tag_id}  center={tag.center[0]:.1f}, {tag.center[1]:.1f}")

            display = cv2.resize(annotated, (_cfg.camera.preview_width, _cfg.camera.preview_height))
            cv2.imshow("Tag Detection", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
