"""
Script to open the webcam and visualize detected ArUco tags.
"""

import cv2

from robot_commander.camera.camera import Camera
from robot_commander.camera.tag_detector import TagDetector, draw_tags


def main():
    detector = TagDetector()

    with Camera(device_index=0, width=1920, height=1080) as cam:
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

            display = cv2.resize(annotated, (960, 540))
            cv2.imshow("Tag Detection", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
