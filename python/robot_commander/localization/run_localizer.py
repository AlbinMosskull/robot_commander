"""
Script to test localization
"""

import cv2

from robot_commander.camera.camera import Camera
from robot_commander.camera.tag_detector import TagDetector, draw_tags
from robot_commander.localization.localizer import Localizer


def main():
    detector = TagDetector()

    _WARMUP_FRAMES = 30

    with Camera(device_index=0) as cam:
        print(f"Warming up camera ({_WARMUP_FRAMES} frames)...")
        for _ in range(_WARMUP_FRAMES):
            cam.read()

        print("Capturing initialization frame...")
        ok, init_frame = cam.read()
        if not ok:
            print("Failed to read initialization frame.")
            return

        localizer = Localizer(detector, init_frame)
        print(f"Origin tag center: {localizer._origin.center}")
        print("Localizer ready. Press 'q' to quit.")

        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            tags = detector.detect(frame)
            annotated = draw_tags(frame, tags)

            pos = localizer.localize(frame)
            if pos is not None:
                x, y = pos
                print(f"  x={x:.1f}  y={y:.1f}")
                cv2.putText(annotated, f"x={x:.1f} y={y:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(annotated, "Cannot localize", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Localization", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
