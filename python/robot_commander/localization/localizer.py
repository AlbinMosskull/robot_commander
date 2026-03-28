import cv2
import numpy as np

from robot_commander.camera.tag_detector import TagDetector


class Localizer:
    """
    Estimates the 3-D position of an AprilTag in the camera frame.

    Uses cv2.solvePnP with the four tag corners to recover the translation
    vector (x, y, z) in metres relative to the camera.

    Args:
        detector: A TagDetector instance.
        camera_matrix: 3×3 intrinsic matrix.
        tag_size: Physical side length of the tag in metres.
        dist_coeffs: Distortion coefficients (defaults to zero distortion).
    """

    # 3-D object points for the tag corners, at unit scale (multiply by tag_size).
    # Order must match DetectedTag.corners: top-left, top-right, bottom-right, bottom-left.
    _UNIT_OBJ_POINTS = np.array([
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [ 0.5,  0.5, 0.0],
        [-0.5,  0.5, 0.0],
    ], dtype=np.float32)

    def __init__(
        self,
        detector: TagDetector,
        camera_matrix: np.ndarray,
        tag_size: float,
        dist_coeffs: np.ndarray | None = None,
    ):
        self._detector = detector
        self._camera_matrix = camera_matrix.astype(np.float64)
        self._dist_coeffs = (
            dist_coeffs.astype(np.float64)
            if dist_coeffs is not None
            else np.zeros(4, dtype=np.float64)
        )
        self._obj_points = self._UNIT_OBJ_POINTS * tag_size

    def localize(self, frame: cv2.typing.MatLike) -> tuple[float, float, float] | None:
        """Return (x, y, z) in metres in the camera frame, or None if no tag found."""
        tags = self._detector.detect(frame)
        if not tags:
            return None

        tag = tags[0]
        success, _, tvec = cv2.solvePnP(
            self._obj_points, tag.corners.astype(np.float32),
            self._camera_matrix, self._dist_coeffs,
        )
        if not success:
            return None

        x, y, z = tvec.flatten()
        return float(x), float(y), float(z)
