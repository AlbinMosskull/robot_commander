"""
Tag-calibrated depth estimation using non-metric Depth Anything V2.

Two AprilTags with known metric depths (from solvePnP) are used to fit an
affine mapping  depth_metric = scale * raw + offset  from the raw relative
depth values produced by the model.
"""

import cv2
import numpy as np

from robot_commander.depth_processing.depth_processor import DepthProcessor
from robot_commander.localization.localizer import Localizer

_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"


class CalibratedDepthProcessor:
    """
    Non-metric Depth Anything V2 calibrated to metric scale via two AprilTags.

    Call calibrate(frame) when two tags are visible to fit the affine mapping.
    Then call process(frame) to get metric depth maps.

    Args:
        localizer: A Localizer instance configured with camera intrinsics and tag size.
    """

    def __init__(self, localizer: Localizer):
        self._base = DepthProcessor(_MODEL)
        self._localizer = localizer
        self._scale: float | None = None
        self._offset: float | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._scale is not None

    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Detect two AprilTags, estimate their metric depths, read back the
        corresponding raw depth-anything values, and fit scale + offset.

        Returns True on success, False if fewer than two tags are found or
        the raw values are too close to distinguish.
        """
        tag_poses = self._localizer.localize_all(frame)
        if len(tag_poses) < 2:
            print(f"Calibration failed: need 2 tags, found {len(tag_poses)}.")
            return False

        raw = self._base.process(frame)

        pairs: list[tuple[float, float]] = []   # (metric_z, raw_avg)
        for tag, (_, _, z) in tag_poses[:2]:
            mask = np.zeros(raw.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [tag.corners.astype(np.int32).reshape(-1, 1, 2)], 1)
            raw_avg = float(raw[mask == 1].mean())
            pairs.append((z, raw_avg))

        (D1, A1), (D2, A2) = pairs
        if abs(A2 - A1) < 1e-6:
            print("Calibration failed: raw depth values of the two tags are too similar.")
            return False

        self._scale = (D2 - D1) / (A2 - A1)
        self._offset = D1 - self._scale * A1
        print(f"Calibration OK — scale={self._scale:.4f}  offset={self._offset:.4f}  "
              f"(D1={D1:.3f}m A1={A1:.3f})  (D2={D2:.3f}m A2={A2:.3f})")
        return True

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a calibrated metric depth map (float32, metres).

        Raises RuntimeError if calibrate() has not been called successfully.
        """
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() before process().")
        raw = self._base.process(frame)
        return (self._scale * raw + self._offset).astype(np.float32)
