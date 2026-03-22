import cv2
import numpy as np

from robot_commander.camera.tag_detector import TagDetector, DetectedTag


class Localizer:
    """
    Localizes a second tag relative to an origin tag.

    The origin tag is the one with the lowest y center in the initialization frame.
    Its position is remembered and used to identify it in subsequent frames by proximity.
    Subsequent calls to localize() return the position of the other tag in the
    coordinate frame of the origin tag (in pixels).

    Args:
        detector: A TagDetector instance.
        init_frame: A frame used to designate the origin tag.
    """

    def __init__(self, detector: TagDetector, init_frame: cv2.typing.MatLike):
        tags = detector.detect(init_frame)
        if not tags:
            raise RuntimeError("No tags found in initialization frame.")
        self._origin: DetectedTag = min(tags, key=lambda t: t.center[1])
        self._detector = detector

    def localize(self, frame: cv2.typing.MatLike) -> tuple[float, float] | None:
        """Return (x, y) of the second tag relative to the origin tag, or None."""
        tags = self._detector.detect(frame)
        if len(tags) < 2:
            return None

        origin = min(
            tags, key=lambda t: np.linalg.norm(np.array(t.center) - np.array(self._origin.center))
        )
        others = [t for t in tags if t is not origin]
        object_to_localize = others[0]  # Assuming one other tag visible

        x = object_to_localize.center[0] - origin.center[0]
        y = object_to_localize.center[1] - origin.center[1]
        return x, y
