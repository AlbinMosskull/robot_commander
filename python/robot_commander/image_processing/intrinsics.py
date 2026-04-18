from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DEFAULT_PATH = Path("calibration/intrinsics.npz")
AGENT_CAMERA_PATH = Path("calibration/agent_camera_intrinsics.npz")


@dataclass
class Intrinsics:
    camera_matrix: np.ndarray   # shape (3, 3)
    dist_coeffs: np.ndarray
    rms_error: float
    image_size: tuple[int, int]  # (width, height)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def crop(self, x1: int, y1: int) -> "Intrinsics":
        """Return intrinsics adjusted for a crop whose top-left corner is (x1, y1).

        Only the principal point (cx, cy) changes — focal lengths are unaffected by
        cropping. The bottom-right corner of the crop is irrelevant: back-projection
        depends only on each pixel's offset from the principal point, not on image bounds.
        """
        matrix = self.camera_matrix.copy()
        matrix[0, 2] -= x1
        matrix[1, 2] -= y1
        return Intrinsics(
            camera_matrix=matrix,
            dist_coeffs=self.dist_coeffs,
            rms_error=self.rms_error,
            image_size=self.image_size,
        )

    def save(self, path: Path) -> None:
        np.savez(
            path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            rms_error=np.array(self.rms_error),
            image_size=np.array(self.image_size),
        )
        print(f"Saved intrinsics to {path}")

    def __str__(self) -> str:
        return (
            f"Camera intrinsics ({self.image_size[0]}x{self.image_size[1]}):\n"
            f"  fx={self.fx:.2f}  fy={self.fy:.2f}  cx={self.cx:.2f}  cy={self.cy:.2f}\n"
            f"  dist_coeffs={self.dist_coeffs.ravel()}\n"
            f"  RMS reprojection error: {self.rms_error:.4f} px"
        )


def load(path: Path = _DEFAULT_PATH) -> Intrinsics:
    data = np.load(path)
    return Intrinsics(
        camera_matrix=data["camera_matrix"],
        dist_coeffs=data["dist_coeffs"],
        rms_error=float(data["rms_error"]),
        image_size=tuple(data["image_size"].tolist()),
    )
