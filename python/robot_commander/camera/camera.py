import cv2


class Camera:
    """
    Wraps a webcam for frame capture.
    
    Args:
        device_index: The index of the camera to use.
                      Find the index by running `ls /dev/video*` on Linux.
    """

    def __init__(self, device_index: int = 0, width: int | None = 1920, height: int | None = 1080):
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera at device index {device_index}")
        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if width is not None or height is not None:
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (width is not None and actual_w != width) or (height is not None and actual_h != height):
                raise RuntimeError(
                    f"Camera does not support resolution {width}x{height}, got {actual_w}x{actual_h}"
                )

    def warm_up(self):
        for _ in range(10):
            self._cap.read()

    def read(self) -> tuple[bool, cv2.typing.MatLike]:
        return self._cap.read()

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
