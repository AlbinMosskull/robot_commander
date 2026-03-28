from dataclasses import dataclass
from pathlib import Path

import numpy as np

# parents[3] from this file: camera/ -> robot_commander/ -> python/ -> repo root
_DEFAULT_PATH = Path(__file__).parents[3] / "intrinsics" / "intrinsics.npz"


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
