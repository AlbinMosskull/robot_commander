import argparse
from pathlib import Path

from robot_commander.agent.adeept.pi_camera import PiCamera
from robot_commander.image_processing.calibration_capture import capture_frames
from robot_commander.image_processing.intrinsics_calibration import calibrate
from robot_commander.config import load as load_config

_cfg = load_config()
_CHECKERBOARD = (_cfg.checkerboard.cols, _cfg.checkerboard.rows)
_SQUARE_SIZE_M = _cfg.checkerboard.square_size_m

_DEFAULT_IMAGES_DIR = Path("images/agent_camera_captured_images")
_DEFAULT_OUTPUT = Path("calibration/agent_camera_intrinsics.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture checkerboard images from the Pi camera and compute intrinsics.")
    parser.add_argument("--images", type=Path, default=_DEFAULT_IMAGES_DIR, help="Directory for calibration images")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Output path for the .npz file")
    parser.add_argument("--skip-capture", action="store_true", help="Skip capture and calibrate from existing images")
    args = parser.parse_args()

    if not args.skip_capture:
        with PiCamera() as cam:
            capture_frames(cam, args.images)

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
