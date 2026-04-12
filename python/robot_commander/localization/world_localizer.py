from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class WorldPose:
    x: float
    y: float
    heading: float


class WorldLocalizer(ABC):
    @abstractmethod
    def localize(self, frame: np.ndarray) -> WorldPose | None: ...