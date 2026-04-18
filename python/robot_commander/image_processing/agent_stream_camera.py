import threading

import cv2
import numpy as np

from robot_commander.image_processing.camera import Camera
from robot_commander.remote_control.agent_client import AgentClient


class AgentStreamCamera(Camera):
    def __init__(self, client: AgentClient):
        self._client = client
        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._frame_available = threading.Event()
        self._thread = threading.Thread(target=self._stream, daemon=True)
        self._thread.start()
        if not self._frame_available.wait(timeout=10.0):
            raise RuntimeError("Timed out waiting for first frame from agent.")

    def _stream(self) -> None:
        for camera_frame_jpg, _, _ in self._client.stream_agent_updates():
            if camera_frame_jpg is not None:
                frame = cv2.imdecode(np.frombuffer(camera_frame_jpg, np.uint8), cv2.IMREAD_COLOR)
                with self._lock:
                    self._latest_frame = frame
                self._frame_available.set()

    def read(self) -> tuple[bool, cv2.typing.MatLike]:
        with self._lock:
            if self._latest_frame is None:
                return False, np.empty((0, 0, 3), dtype=np.uint8)
            return True, self._latest_frame.copy()

    def release(self) -> None:
        self._client.close()
