from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectedTag:
    tag_id: int
    corners: np.ndarray  # shape (4, 2), pixel coordinates of the four corners
    center: tuple[float, float]


class TagDetector:
    """Detects ArUco tags in frames.

    Args:
        dictionary: ArUco dictionary ID.
    """

    def __init__(self, dictionary: int = cv2.aruco.DICT_4X4_50):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def detect(self, frame: cv2.typing.MatLike) -> list[DetectedTag]:
        corners, ids, _ = self._detector.detectMarkers(frame)
        if ids is None:
            return []

        results = []
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            pts = tag_corners[0]
            center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
            results.append(DetectedTag(tag_id=int(tag_id), corners=pts, center=center))
        return results

    def draw(self, frame: cv2.typing.MatLike, tags: list[DetectedTag]) -> cv2.typing.MatLike:
        out = frame.copy()
        for tag in tags:
            pts = tag.corners.astype(int)
            cv2.polylines(out, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
            cx, cy = int(tag.center[0]), int(tag.center[1])
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(out, f"ID {tag.tag_id}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out