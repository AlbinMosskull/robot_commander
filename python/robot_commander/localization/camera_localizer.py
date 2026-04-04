import numpy as np

from robot_commander.localization.localizer import Localizer
from robot_commander.localization.world_localizer import WorldLocalizer
from robot_commander.map_building.map_coordinates import MapCoordinates


class CameraLocalizer(WorldLocalizer):
    def __init__(self, localizer: Localizer, map_coords: MapCoordinates):
        self._localizer = localizer
        self._map_coords = map_coords

    def localize(self, frame: np.ndarray) -> tuple[float, float] | None:
        result = self._localizer.localize(frame)
        if result is None:
            return None
        x, y, z = result
        return self._map_coords.camera_to_world_2d(x, y, z)
