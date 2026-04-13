import cv2
import numpy as np

from robot_commander.localization.localizer import Localizer
from robot_commander.localization.world_localizer import WorldLocalizer, WorldPose
from robot_commander.map.map_coordinates import MapCoordinates


def _heading_from_rvec(rvec: np.ndarray, u_floor: np.ndarray, v_floor: np.ndarray) -> float:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    tag_x_in_camera = rotation_matrix @ np.array([1.0, 0.0, 0.0])
    return float(np.arctan2(tag_x_in_camera @ v_floor, tag_x_in_camera @ u_floor))


class CameraLocalizer(WorldLocalizer):
    def __init__(self, localizer: Localizer, map_coords: MapCoordinates, heading_offset: float = 0.0):
        self._localizer = localizer
        self._map_coords = map_coords
        self._heading_offset = heading_offset

    def localize(self, frame: np.ndarray) -> WorldPose | None:
        results = self._localizer.localize_all(frame)
        if not results:
            return None
        _, (x, y, z), rvec = results[0]
        world_x, world_y = self._map_coords.camera_to_world_2d(x, y, z)
        raw_heading = _heading_from_rvec(rvec, self._map_coords.u_floor, self._map_coords.v_floor)
        heading = float(np.arctan2(
            np.sin(raw_heading + self._heading_offset),
            np.cos(raw_heading + self._heading_offset),
        ))
        return WorldPose(world_x, world_y, heading)
