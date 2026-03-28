"""
Tag-calibrated depth estimation using non-metric Depth Anything V2.

Two AprilTags with known metric depths are used to fit an
affine mapping  depth_metric = scale * raw + offset  from the raw relative
depth values produced by the model.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robot_commander.depth_processing.depth_processor import DepthProcessor
from robot_commander.localization.localizer import Localizer

_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
_DEFAULT_CALIBRATION_PATH = Path("calibration/depth_calibration.npz")


@dataclass
class DepthCalibration:
    scale: float
    offset: float

    def save(self, path: Path = _DEFAULT_CALIBRATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, scale=self.scale, offset=self.offset)
        print(f"Saved depth calibration to {path}")

    @classmethod
    def load(cls, path: Path = _DEFAULT_CALIBRATION_PATH) -> "DepthCalibration":
        data = np.load(path)
        return cls(scale=float(data["scale"]), offset=float(data["offset"]))


class CalibratedDepthProcessor:
    """
    Applies an affine calibration to raw Depth Anything V2 output to produce
    metric depth maps.

    Construct via from_calibration() to load a saved calibration, or call
    calibrate() after construction to fit one from a live frame.
    """

    def __init__(self) -> None:
        self._base = DepthProcessor(_MODEL)
        self._calibration: DepthCalibration | None = None

    @classmethod
    def from_calibration(cls, calibration: DepthCalibration) -> "CalibratedDepthProcessor":
        processor = cls()
        processor._calibration = calibration
        return processor

    @property
    def is_calibrated(self) -> bool:
        return self._calibration is not None

    def calibrate(
        self, frame: np.ndarray, localizer: Localizer
    ) -> tuple[DepthCalibration, np.ndarray] | None:
        """
        Detect two AprilTags, fit the affine scale+offset, and return the
        resulting calibration and the calibrated depth map for this frame.

        Returns None if fewer than two tags are found or the raw values are
        too close to distinguish.
        """
        tag_poses = localizer.localize_all(frame)
        if len(tag_poses) < 2:
            print(f"Calibration failed: need 2 tags, found {len(tag_poses)}.")
            return None

        raw = self._base.process(frame)

        pairs: list[tuple[float, float]] = []
        for tag, (_, _, z) in tag_poses[:2]:
            mask = np.zeros(raw.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [tag.corners.astype(np.int32).reshape(-1, 1, 2)], 1)
            raw_avg = float(raw[mask == 1].mean())
            pairs.append((z, raw_avg))

        (D1, A1), (D2, A2) = pairs
        if abs(A2 - A1) < 1e-6:
            print("Calibration failed: raw depth values of the two tags are too similar.")
            return None

        scale = (D2 - D1) / (A2 - A1)
        offset = D1 - scale * A1
        print(f"Calibration OK — scale={scale:.4f}  offset={offset:.4f}  "
              f"(D1={D1:.3f}m A1={A1:.3f})  (D2={D2:.3f}m A2={A2:.3f})")

        self._calibration = DepthCalibration(scale=scale, offset=offset)

        calibrated = scale * raw + offset
        calibrated[raw == 0] = 0.0
        return self._calibration, calibrated.astype(np.float32)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a calibrated metric depth map (float32, metres).

        Raises RuntimeError if calibrate() has not been called successfully.
        """
        if self._calibration is None:
            raise RuntimeError("Call calibrate() before process().")
        raw = self._base.process(frame)
        calibrated = self._calibration.scale * raw + self._calibration.offset
        calibrated[raw == 0] = 0.0
        return calibrated.astype(np.float32)
