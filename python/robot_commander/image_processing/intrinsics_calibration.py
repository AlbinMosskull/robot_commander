from pathlib import Path

import cv2
import numpy as np

from robot_commander.image_processing.intrinsics import Intrinsics


def calibrate(
    image_paths: list[Path],
    checkerboard: tuple[int, int],
    square_size: float,
) -> Intrinsics:
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


def _object_points(checkerboard: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = checkerboard
    pts = np.zeros((rows * cols, 3), dtype=np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return pts
