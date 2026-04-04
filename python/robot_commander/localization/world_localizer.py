from abc import ABC, abstractmethod

import numpy as np


class WorldLocalizer(ABC):
    @abstractmethod
    def localize(self, frame: np.ndarray) -> tuple[float, float] | None:
        """Return (x, y) in world coordinates, or None if localization failed."""
