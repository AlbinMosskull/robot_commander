from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationResult:
    label: str
    score: float
    mask: np.ndarray  # bool (H, W)


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def overlaps(self, other: "BoundingBox") -> bool:
        return self.x1 <= other.x2 and other.x1 <= self.x2 and self.y1 <= other.y2 and other.y1 <= self.y2

    def merge(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "BoundingBox | None":
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return cls(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
