import numpy as np

from robot_commander.image_processing.camera import Camera
from robot_commander.remote_control.controller import RemoteControl


class AgentCamera(Camera):
    def __init__(self, controller: RemoteControl):
        self._controller = controller

    def read(self) -> tuple[bool, np.ndarray]:
        frame = self._controller.latest_agent_frame
        if frame is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        return True, frame
