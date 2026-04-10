import argparse
import datetime
from pathlib import Path

import cv2

from robot_commander.image_processing.camera import WebCamera
from robot_commander.image_processing.intrinsics_calibration import calibrate
from robot_commander.config import load as load_config

_cfg = load_config()
_CHECKERBOARD = (_cfg.checkerboard.cols, _cfg.checkerboard.rows)
_SQUARE_SIZE_M = _cfg.checkerboard.square_size_m

_DEFAULT_IMAGES_DIR = Path("images/captured_images")
_DEFAULT_OUTPUT = Path("calibration/intrinsics.npz")


def _capture(save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    with WebCamera() as cam:
        print("Camera opened.")
        print("  SPACE       - capture frame")
        print("  y / ENTER   - approve and save")
        print("  n / ESC     - deny, back to live view")
        print("  q           - quit and proceed to calibration")

        captured_frame = None

        while True:
            if captured_frame is None:
                ok, frame = cam.read()
                if not ok:
                    print("Failed to read frame.")
                    break
                display = cv2.resize(frame.copy(), (_cfg.camera.preview_width, _cfg.camera.preview_height))
                cv2.putText(display, "Press SPACE to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Capture Image", display)
            else:
                display = cv2.resize(captured_frame.copy(), (_cfg.camera.preview_width, _cfg.camera.preview_height))
                cv2.putText(display, "Approve? y=yes  n=no", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.imshow("Capture Image", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" ") and captured_frame is None:
                _, captured_frame = cam.read()
                print("Frame captured. Approve (y/ENTER) or deny (n/ESC)?")
            elif captured_frame is not None and key in (ord("y"), 13):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = save_dir / f"image_{timestamp}.png"
                cv2.imwrite(str(path), captured_frame)
                print(f"Saved: {path}")
                captured_frame = None
            elif captured_frame is not None and key in (ord("n"), 27):
                print("Denied. Returning to live view.")
                captured_frame = None

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture checkerboard images and compute camera intrinsics.")
    parser.add_argument("--images", type=Path, default=_DEFAULT_IMAGES_DIR, help="Directory for calibration images")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Output path for the .npz file")
    parser.add_argument("--skip-capture", action="store_true", help="Skip capture and calibrate from existing images")
    args = parser.parse_args()

    if not args.skip_capture:
        _capture(args.images)

    image_paths = sorted(args.images.glob("*.png")) + sorted(args.images.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No PNG/JPG images found in {args.images}")

    print(f"\nFound {len(image_paths)} image(s) in {args.images}")
    intrinsics = calibrate(image_paths, checkerboard=_CHECKERBOARD, square_size=_SQUARE_SIZE_M)

    print()
    print(intrinsics)
    intrinsics.save(args.output)


if __name__ == "__main__":
    main()
