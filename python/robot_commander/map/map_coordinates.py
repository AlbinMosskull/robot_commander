from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_MAP_SCALE = 150
_MAP_W, _MAP_H = 600, 600
_MAP_ORIGIN = (300, 540)


@dataclass
class MapCoordinates:
    scale_px_per_m: float
    width_px: int
    height_px: int
    origin_px: tuple[int, int]
    background: np.ndarray | None = field(default=None, repr=False)
    u_floor: np.ndarray | None = field(default=None, repr=False)
    v_floor: np.ndarray | None = field(default=None, repr=False)

    def to_map_px(self, coords_2d: np.ndarray) -> np.ndarray:
        ox, oy = self.origin_px
        return np.column_stack([
            (ox + coords_2d[:, 0] * self.scale_px_per_m).astype(np.int32),
            (oy - coords_2d[:, 1] * self.scale_px_per_m).astype(np.int32),
        ])

    def px_to_world(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        ox, oy = self.origin_px
        return (pixel_x - ox) / self.scale_px_per_m, (oy - pixel_y) / self.scale_px_per_m

    def world_to_px(self, x: float, y: float) -> tuple[int, int]:
        pts = self.to_map_px(np.array([[x, y]]))
        return int(pts[0, 0]), int(pts[0, 1])

    def camera_to_world_2d(self, x: float, y: float, z: float) -> tuple[float, float]:
        if self.u_floor is None or self.v_floor is None:
            raise RuntimeError("Floor basis vectors not available — was the map built with a real camera?")
        point = np.array([x, y, z])
        return float(point @ self.u_floor), float(point @ self.v_floor)

    def save(self, path: Path, image_filename: str = "stencil_map.png") -> None:
        if self.background is not None:
            cv2.imwrite(str(path.parent / image_filename), self.background)
        data = {
            "image": image_filename,
            "scale_px_per_m": self.scale_px_per_m,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "origin_px": list(self.origin_px),
        }
        if self.u_floor is not None:
            data["u_floor"] = self.u_floor.tolist()
        if self.v_floor is not None:
            data["v_floor"] = self.v_floor.tolist()
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> MapCoordinates:
        data = json.loads(path.read_text())
        image_path = path.parent / data["image"]
        background = cv2.imread(str(image_path))
        if background is None:
            background = np.full((data["height_px"], data["width_px"], 3), 30, dtype=np.uint8)
        u_floor = np.array(data["u_floor"]) if "u_floor" in data else None
        v_floor = np.array(data["v_floor"]) if "v_floor" in data else None
        return cls(
            scale_px_per_m=data["scale_px_per_m"],
            width_px=data["width_px"],
            height_px=data["height_px"],
            origin_px=tuple(data["origin_px"]),
            background=background,
            u_floor=u_floor,
            v_floor=v_floor,
        )

    @classmethod
    def default(cls) -> MapCoordinates:
        background = np.full((_MAP_H, _MAP_W, 3), 30, dtype=np.uint8)
        return cls(
            scale_px_per_m=_MAP_SCALE,
            width_px=_MAP_W,
            height_px=_MAP_H,
            origin_px=_MAP_ORIGIN,
            background=background,
        )
