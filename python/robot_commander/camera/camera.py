import cv2


class Camera:
    """
    Wraps a webcam for frame capture.
    
    Args:
        device_index: The index of the camera to use.
                      Find the index by running `ls /dev/video*` on Linux.
    """

    def __init__(self, device_index: int = 0):
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera at device index {device_index}")

    def read(self) -> tuple[bool, cv2.typing.MatLike]:
        return self._cap.read()

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
