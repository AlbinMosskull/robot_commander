from dataclasses import dataclass
from pathlib import Path

import numpy as np

# parents[3] from this file: calibration/ -> robot_commander/ -> python/ -> repo root
_DEFAULT_PATH = Path(__file__).parents[3] / "intrinsics" / "intrinsics.npz"


@dataclass(frozen=True)
class Intrinsics:
    camera_matrix: np.ndarray  # 3×3
    dist_coeffs: np.ndarray

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


def load(path: Path = _DEFAULT_PATH) -> Intrinsics:
    data = np.load(path)
    return Intrinsics(
        camera_matrix=data["camera_matrix"],
        dist_coeffs=data["dist_coeffs"],
    )
