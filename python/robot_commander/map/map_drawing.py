import cv2
import numpy as np

from robot_commander.image_processing import intrinsics as cal
from robot_commander.map.map_coordinates import MapCoordinates


def _shadow_polygon(
    contour: np.ndarray,
    camera_height: float,
    surface_height: float,
    map_coords: MapCoordinates,
) -> np.ndarray:
    s = camera_height / (camera_height - surface_height)
    ox, oy = map_coords.origin_px
    pts = contour[:, 0, :].astype(float)
    shadow = np.column_stack([
        ox + s * (pts[:, 0] - ox),
        oy + s * (pts[:, 1] - oy),
    ]).astype(np.int32)
    return shadow.reshape(-1, 1, 2)


def _draw_camera_symbol(canvas: np.ndarray, intr: cal.Intrinsics, map_coords: MapCoordinates) -> None:
    near_z, far_z = 0.05, 0.2
    near_hw = near_z * intr.cx / intr.fx
    far_hw  = far_z  * intr.cx / intr.fx
    pts_2d = np.array([
        (-near_hw, near_z), (near_hw, near_z),
        ( far_hw,  far_z),  (-far_hw, far_z),
    ], dtype=np.float32)
    px = map_coords.to_map_px(pts_2d).reshape(-1, 1, 2)
    cv2.polylines(canvas, [px], isClosed=True, color=(180, 60, 0), thickness=2)
    ox, oy = map_coords.origin_px
    cv2.putText(canvas, "Camera", (ox + 12, oy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 60, 0), 1)


def draw_stencil_map(
    footprints: dict[str, np.ndarray],
    intr: cal.Intrinsics,
    camera_height: float,
    surface_heights: dict[str, float],
    map_coords: MapCoordinates,
) -> np.ndarray:
    canvas = np.full((map_coords.height_px, map_coords.width_px, 3), 255, dtype=np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    smoothed: list[tuple[str, list]] = []
    for label, coords_2d in footprints.items():
        px = map_coords.to_map_px(coords_2d)
        valid = (
            (px[:, 0] >= 0) & (px[:, 0] < map_coords.width_px) &
            (px[:, 1] >= 0) & (px[:, 1] < map_coords.height_px)
        )
        if valid.sum() < 3:
            continue
        cell = np.zeros((map_coords.height_px, map_coords.width_px), dtype=np.uint8)
        cell[px[valid, 1], px[valid, 0]] = 1
        cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            smoothed.append((label, contours))

    for label, contours in smoothed:
        if label not in surface_heights:
            continue
        h_surface = surface_heights[label]
        if camera_height <= h_surface:
            continue
        largest = max(contours, key=cv2.contourArea)
        shadow = _shadow_polygon(largest, camera_height, h_surface, map_coords)
        cv2.fillPoly(canvas, [shadow], (210, 210, 225))

    for label, contours in smoothed:
        cv2.drawContours(canvas, contours, -1, color=(0, 0, 0), thickness=2)
        largest = max(contours, key=cv2.contourArea)
        lx = int(largest[:, 0, 0].mean())
        ly = int(largest[:, 0, 1].min()) - 8
        cv2.putText(canvas, label.capitalize(), (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    _draw_camera_symbol(canvas, intr, map_coords)
    return canvas
