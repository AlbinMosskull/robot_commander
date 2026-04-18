import cv2
import numpy as np
from picamera2 import Picamera2

from robot_commander.image_processing.camera import Camera


class PiCamera(Camera):
    def __init__(self, width: int = 1920, height: int = 1080):
        self._camera = Picamera2()
        config = self._camera.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._camera.configure(config)
        self._camera.start()

    def read(self) -> tuple[bool, cv2.typing.MatLike]:
        frame = self._camera.capture_array()
        if frame is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        self._camera.stop()
