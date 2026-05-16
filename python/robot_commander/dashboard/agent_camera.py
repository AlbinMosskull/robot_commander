import cv2
import numpy as np

from robot_commander.image_processing.camera import Camera
from robot_commander.remote_control.controller import RemoteControl


def _signal_lost_frame(reference: np.ndarray | None) -> np.ndarray:
    h, w = reference.shape[:2] if reference is not None else (360, 640)
    canvas = np.full((h, w, 3), 13, dtype=np.uint8)
    text = "SIGNAL LOST"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(canvas, text, ((w - tw) // 2, (h + th) // 2), font, scale, (60, 60, 180), thickness)
    return canvas


class OverheadCamera(Camera):
    def __init__(self, controller: RemoteControl):
        self._controller = controller

    def read(self) -> tuple[bool, np.ndarray]:
        frame = self._controller.latest_overhead_frame
        if frame is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        return True, frame


class AgentCamera(Camera):
    def __init__(self, controller: RemoteControl):
        self._controller = controller

    def read(self) -> tuple[bool, np.ndarray]:
        frame = self._controller.latest_agent_frame
        if self._controller.connection_lost:
            return True, _signal_lost_frame(frame)
        if frame is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        return True, frame
