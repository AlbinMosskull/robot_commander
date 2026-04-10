"""
This module is responsible for the geometric processing of building the stencil map.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from robot_commander.image_processing import intrinsics as cal
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud
from robot_commander.depth_processing.ransac import detect_planes, Plane
from robot_commander.map.map_coordinates import MapCoordinates

_MIN_OBJECT_HEIGHT = 0.10
_MAX_OBJECT_HEIGHT = 1.50
_MAX_SURFACE_TILT_DEG = 40


@dataclass
class FootprintResult:
    footprints: dict[str, np.ndarray]
    surface_heights: dict[str, float]
    u_floor: np.ndarray
    v_floor: np.ndarray
    label_points: dict[str, np.ndarray]


def _floor_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (u=right, v=forward) unit vectors lying in the floor plane."""
    x_cam = np.array([1.0, 0.0, 0.0])
    u = x_cam - (x_cam @ normal) * normal
    if np.linalg.norm(u) < 1e-6:
        x_cam = np.array([0.0, 0.0, 1.0])
        u = x_cam - (x_cam @ normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def to_floor_2d(points: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project 3D camera-space points to (right, forward) floor coordinates."""
    return np.column_stack([points @ u, points @ v])


def _filter_above_floor(
    points: np.ndarray, n_floor: np.ndarray, d_floor: float
) -> np.ndarray:
    """Return boolean mask of points within the plausible furniture height range."""
    heights = points @ n_floor - d_floor
    return (heights > _MIN_OBJECT_HEIGHT) & (heights < _MAX_OBJECT_HEIGHT)


def _largest_floor_component(
    pts: np.ndarray, u: np.ndarray, v: np.ndarray, map_coords: MapCoordinates
) -> np.ndarray:
    """Return boolean mask keeping only the largest connected cluster in floor 2D space."""
    coords = to_floor_2d(pts, u, v)
    px = map_coords.to_map_px(coords)

    canvas = np.zeros((map_coords.height_px, map_coords.width_px), dtype=np.uint8)
    valid = (
        (px[:, 0] >= 0) & (px[:, 0] < map_coords.width_px) &
        (px[:, 1] >= 0) & (px[:, 1] < map_coords.height_px)
    )
    canvas[px[valid, 1], px[valid, 0]] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(canvas, kernel)

    n_labels, labels = cv2.connectedComponents(dilated)
    if n_labels <= 1:
        return valid

    comp_ids = labels[px[valid, 1], px[valid, 0]]
    counts = np.bincount(comp_ids, minlength=n_labels)
    counts[0] = 0
    largest = int(counts.argmax())

    keep = np.zeros(len(pts), dtype=bool)
    keep[np.where(valid)[0][comp_ids == largest]] = True
    return keep


def _resize_mask_to(mask: np.ndarray, shape: tuple) -> np.ndarray:
    h, w = shape[:2]
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def detect_floor(
    depth: np.ndarray,
    intrinsics: cal.Intrinsics,
) -> Plane | None:
    """Detect the dominant floor plane. Returns a Plane with normal pointing toward camera."""
    pts = depth_image_to_point_cloud(depth, intrinsics)
    planes = detect_planes(pts, n_planes=1, n_iterations=500, distance_threshold=0.03)
    if not planes:
        return None
    plane = planes[0]
    if plane.distance > 0:
        plane = Plane(normal=-plane.normal, distance=-plane.distance, inliers=plane.inliers)
    return plane


def build_footprints(
    depths: list[np.ndarray],
    frame_masks: dict[str, np.ndarray],
    intrinsics: cal.Intrinsics,
    n_floor: np.ndarray,
    d_floor: float,
    map_coords: MapCoordinates,
) -> FootprintResult:
    """Project segmented objects onto the floor plane across all frames."""
    u, v = _floor_basis(n_floor)

    class_2d: dict[str, np.ndarray] = {}
    surface_heights: dict[str, float] = {}
    label_points: dict[str, np.ndarray] = {}

    for label, frame_mask in frame_masks.items():
        pts_list = []
        for depth in depths:
            mask = _resize_mask_to(frame_mask, depth.shape)
            md = depth.copy()
            md[~mask] = 0.0
            pts = depth_image_to_point_cloud(md, intrinsics)
            if len(pts):
                pts_list.append(pts)

        if not pts_list:
            continue

        merged = np.vstack(pts_list)
        above = _filter_above_floor(merged, n_floor, d_floor)
        filtered = merged[above]
        if len(filtered) < 10:
            continue

        planes = detect_planes(filtered, n_planes=1, n_iterations=300,
                               distance_threshold=0.06)
        if planes:
            tilt = float(np.degrees(np.arccos(
                np.clip(abs(float(np.dot(planes[0].normal, n_floor))), 0.0, 1.0)
            )))
            if tilt <= _MAX_SURFACE_TILT_DEG:
                filtered = filtered[planes[0].inliers]

        comp_mask = _largest_floor_component(filtered, u, v, map_coords)
        filtered = filtered[comp_mask]
        if len(filtered) >= 3:
            class_2d[label] = to_floor_2d(filtered, u, v)
            surface_heights[label] = float(np.mean(filtered @ n_floor - d_floor))
            label_points[label] = filtered

    return FootprintResult(
        footprints=class_2d,
        surface_heights=surface_heights,
        u_floor=u,
        v_floor=v,
        label_points=label_points,
    )
