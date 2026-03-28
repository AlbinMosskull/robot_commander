"""
Script to capture images from the webcam with approve/deny flow.
Useful for collecting images for calibration.

Controls:
  SPACE  - capture a frame
  y / ENTER - approve and save the captured frame
  n / ESC   - deny and return to live view
  q      - quit
"""

import datetime
from pathlib import Path

import cv2

from robot_commander.image_processing.camera import Camera
from robot_commander.config import load as load_config

SAVE_DIR = Path("images/captured_images")
_cfg = load_config()


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    with Camera() as cam:
        print("Camera opened.")
        print("  SPACE       - capture frame")
        print("  y / ENTER   - approve and save")
        print("  n / ESC     - deny, back to live view")
        print("  q           - quit")

        captured_frame = None

        while True:
            if captured_frame is None:
                ok, frame = cam.read()
                if not ok:
                    print("Failed to read frame.")
                    break
                display = cv2.resize(frame.copy(), (_cfg.camera.preview_width, _cfg.camera.preview_height))
                cv2.putText(
                    display,
                    "Press SPACE to capture",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Capture Image", display)
            else:
                display = cv2.resize(captured_frame.copy(), (_cfg.camera.preview_width, _cfg.camera.preview_height))
                cv2.putText(
                    display,
                    "Approve? y=yes  n=no",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                )
                cv2.imshow("Capture Image", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" ") and captured_frame is None:
                _, captured_frame = cam.read()
                print("Frame captured. Approve (y/ENTER) or deny (n/ESC)?")
            elif captured_frame is not None and key in (ord("y"), 13):  # y or ENTER
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = SAVE_DIR / f"image_{timestamp}.png"
                cv2.imwrite(str(path), captured_frame)
                print(f"Saved: {path}")
                captured_frame = None
            elif captured_frame is not None and key in (ord("n"), 27):  # n or ESC
                print("Denied. Returning to live view.")
                captured_frame = None

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()