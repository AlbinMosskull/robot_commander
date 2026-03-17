"""Run this file to open the webcam and visualize detected ArUco tags.

Press 'q' to quit.
"""

import cv2

from robot_commander.camera.camera import Camera
from robot_commander.camera.tag_detector import TagDetector


def main():
    detector = TagDetector()

    with Camera(device_index=0) as cam:
        print("Camera opened. Press 'q' to quit.")
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            tags = detector.detect(frame)
            annotated = detector.draw(frame, tags)

            for tag in tags:
                print(f"  Tag ID={tag.tag_id}  center={tag.center[0]:.1f}, {tag.center[1]:.1f}")

            cv2.imshow("Tag Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

