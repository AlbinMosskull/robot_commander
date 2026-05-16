import numpy as np

from robot_commander.dashboard.signal_lost import signal_lost_frame
from robot_commander.image_processing.camera import Camera
from robot_commander.remote_control.controller import RemoteControl


def _signal_lost_frame(reference: np.ndarray | None) -> np.ndarray:
    h, w = reference.shape[:2] if reference is not None else (360, 640)
    return signal_lost_frame(w, h)


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
