"""
Compute camera intrinsics from checkerboard calibration images.

Usage:
    uv run python -m robot_commander.camera.calibrate_intrinsics
    uv run python -m robot_commander.camera.calibrate_intrinsics --images path/to/images --output intrinsics.npz

The script detects a checkerboard pattern in each image, runs OpenCV's
calibrateCamera, and saves the resulting intrinsic matrix and distortion
coefficients to a .npz file.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from robot_commander.camera.intrinsics import Intrinsics
from robot_commander.config import load as load_config

_cfg = load_config()
CHECKERBOARD = (_cfg.checkerboard.cols, _cfg.checkerboard.rows)
SQUARE_SIZE_M = _cfg.checkerboard.square_size_m


def _object_points(checkerboard: tuple[int, int], square_size: float) -> np.ndarray:
    """3-D coordinates of checkerboard corners in the board frame."""
    cols, rows = checkerboard
    pts = np.zeros((rows * cols, 3), dtype=np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return pts


def calibrate(
    image_paths: list[Path],
    checkerboard: tuple[int, int] = CHECKERBOARD,
    square_size: float = SQUARE_SIZE_M,
) -> Intrinsics:
    """
    Run checkerboard calibration on a list of images.

    Args:
        image_paths: Paths to calibration images.
        checkerboard: Inner corner count (cols, rows).
        square_size: Physical side length of one square in metres.

    Returns:
        CameraIntrinsics with the computed camera matrix and distortion coefficients.
    """
    objp = _object_points(checkerboard, square_size)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [skip] could not read {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        if not found:
            print(f"  [skip] checkerboard not found in {path.name}")
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        print(f"  [ok]   {path.name}")

    if len(objpoints) < 3:
        raise RuntimeError(
            f"Need at least 3 images with a detected checkerboard, got {len(objpoints)}."
        )

    assert image_size is not None
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    return Intrinsics(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms_error=rms,
        image_size=image_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute camera intrinsics from checkerboard images.")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("images/captured_images"),
        help="Directory containing calibration images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("intrinsics.npz"),
        help="Output path for the .npz file",
    )
    args = parser.parse_args()

    image_dir: Path = args.images
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {image_dir}")

    image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No PNG/JPG images found in {image_dir}")

    print(f"Found {len(image_paths)} image(s) in {image_dir}")
    intrinsics = calibrate(image_paths, checkerboard=CHECKERBOARD, square_size=SQUARE_SIZE_M)
    
    print()
    print(intrinsics)
    intrinsics.save(args.output)


if __name__ == "__main__":
    main()
