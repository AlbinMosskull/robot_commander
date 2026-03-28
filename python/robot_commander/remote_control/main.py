"""
This script should produce a top-down stencil map

The goal is to have something similar to "goal stencile map.png" in the repo root.

The algorithm should be the following:

1. Warm up the camera and capture a single frame.
2. Create a calibrated depth processor
3. Compute a depth map from the calibrated depth processor
4. Compute a point cloud, and use ransac to find the dominant plane (the floor)
5. Run semantic segmentation and pick out the classes dining table and couch.
6. For the semantic classes, project all visible points onto the floor plane
7. Save this as a stencil map, showing the objects and the camera position. Also save the original image.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from robot_commander.image_processing import intrinsics as cal
from robot_commander.image_processing.camera import Camera
from robot_commander.config import load as load_config
from robot_commander.image_processing.tag_detector import TagDetector
from robot_commander.depth_processing.calibrated_depth_processor import CalibratedDepthProcessor
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud
from robot_commander.depth_processing.ransac import detect_planes
from robot_commander.localization.localizer import Localizer
from robot_commander.semantic_understanding.semantic_segmentor import SemanticSegmentor

# Maps every model label we care about to a canonical display name.
# Mask2Former/COCO uses "diningtable" (no space) and sometimes "sofa" for couch.
_OBJECT_CLASSES: dict[str, str] = {
    "dining table": "dining table",
    "couch":        "couch",
    # "chair":        "chair",
}
_cfg = load_config()
_NUM_FRAMES = 5          # frames to capture for segmentation coverage
_FRAME_INTERVAL_S = 0.3  # seconds between captures
_MAP_SCALE = 150         # pixels per metre
_MAP_W, _MAP_H = 600, 600
_MAP_ORIGIN = (300, 540) # camera position in map pixels (x, y from top-left)
_OUTPUT_DIR = Path("output")
_MIN_OBJECT_HEIGHT = 0.10  # metres above floor — ignore points below this (floor noise)
_MAX_OBJECT_HEIGHT = 1.50  # metres above floor — ignore points above this (ceiling / walls)
_MAX_SURFACE_TILT_DEG = 40  # reject RANSAC planes tilted more than this from horizontal


# ── Calibration (auto) ─────────────────────────────────────────────────────────

def _auto_calibrate(
    cam: Camera, processor: CalibratedDepthProcessor, detector: TagDetector
) -> np.ndarray | None:
    """Silently calibrate as soon as 2 AprilTags are detected. No window shown."""
    while True:
        ok, frame = cam.read()
        if not ok:
            return None
        tags = detector.detect(frame)
        n = len(tags)
        print(f"  {n}/2 tags visible...", end="\r")
        if n >= 2 and processor.calibrate(frame):
            print()
            return frame


# ── Floor coordinate system ────────────────────────────────────────────────────

def _floor_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (u=right, v=forward) unit vectors lying in the floor plane.

    For any 3D camera-space point p, its top-down floor coordinate is simply
    (p·u, p·v) — the floor-normal component drops out since u,v ⊥ normal.
    """
    x_cam = np.array([1.0, 0.0, 0.0])
    u = x_cam - (x_cam @ normal) * normal
    if np.linalg.norm(u) < 1e-6:
        x_cam = np.array([0.0, 0.0, 1.0])
        u = x_cam - (x_cam @ normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def _to_floor_2d(points: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project 3D camera-space points to (right, forward) floor coordinates."""
    return np.column_stack([points @ u, points @ v])


def _filter_above_floor(
    points: np.ndarray, n_floor: np.ndarray, d_floor: float
) -> np.ndarray:
    """Return boolean mask of points within the plausible furniture height range."""
    heights = points @ n_floor - d_floor
    return (heights > _MIN_OBJECT_HEIGHT) & (heights < _MAX_OBJECT_HEIGHT)


def _largest_floor_component(pts: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return boolean mask keeping only the largest connected cluster in floor 2D space."""
    coords = _to_floor_2d(pts, u, v)
    px = _to_map_px(coords)

    canvas = np.zeros((_MAP_H, _MAP_W), dtype=np.uint8)
    valid = (
        (px[:, 0] >= 0) & (px[:, 0] < _MAP_W) &
        (px[:, 1] >= 0) & (px[:, 1] < _MAP_H)
    )
    canvas[px[valid, 1], px[valid, 0]] = 1

    # Dilate to bridge gaps between sparse projected points
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(canvas, kernel)

    n_labels, labels = cv2.connectedComponents(dilated)
    if n_labels <= 1:
        return valid

    comp_ids = labels[px[valid, 1], px[valid, 0]]
    counts = np.bincount(comp_ids, minlength=n_labels)
    counts[0] = 0  # ignore background
    largest = int(counts.argmax())

    keep = np.zeros(len(pts), dtype=bool)
    keep[np.where(valid)[0][comp_ids == largest]] = True
    return keep


def _to_map_px(coords_2d: np.ndarray) -> np.ndarray:
    ox, oy = _MAP_ORIGIN
    return np.column_stack([
        (ox + coords_2d[:, 0] * _MAP_SCALE).astype(np.int32),
        (oy - coords_2d[:, 1] * _MAP_SCALE).astype(np.int32),
    ])


# ── Stencil map drawing ────────────────────────────────────────────────────────

def _table_shadow_polygon(
    contour: np.ndarray,
    camera_height: float,
    table_height: float,
) -> np.ndarray:
    """
    Compute the floor shadow of the table using 3-D geometry.

    A floor point F is in shadow when the ray from the camera (at height
    camera_height above the floor) through F passes through the table surface
    (at height table_height).  This is equivalent to scaling the table's
    footprint polygon outward from the camera's floor position by:

        s = camera_height / (camera_height - table_height)

    In map-pixel space the camera sits at _MAP_ORIGIN, so the scaling is
    applied relative to that point.

    Args:
        contour:       Smoothed footprint contour, shape (N, 1, 2) from
                       cv2.findContours / cv2.convexHull.
        camera_height: Camera height above the floor in metres.
        table_height:  Table surface height above the floor in metres.

    Returns:
        (N, 1, 2) int32 array suitable for cv2.fillPoly.
    """
    s = camera_height / (camera_height - table_height)
    ox, oy = _MAP_ORIGIN
    pts = contour[:, 0, :].astype(float)          # (N, 2)
    shadow = np.column_stack([
        ox + s * (pts[:, 0] - ox),
        oy + s * (pts[:, 1] - oy),
    ]).astype(np.int32)
    return shadow.reshape(-1, 1, 2)


def _draw_camera_symbol(canvas: np.ndarray, intr: cal.Intrinsics) -> None:
    near_z, far_z = 0.05, 0.2
    near_hw = near_z * intr.cx / intr.fx
    far_hw  = far_z  * intr.cx / intr.fx
    pts_2d = np.array([
        (-near_hw, near_z), (near_hw, near_z),
        ( far_hw,  far_z),  (-far_hw, far_z),
    ], dtype=np.float32)
    px = _to_map_px(pts_2d).reshape(-1, 1, 2)
    cv2.polylines(canvas, [px], isClosed=True, color=(180, 60, 0), thickness=2)
    ox, oy = _MAP_ORIGIN
    cv2.putText(canvas, "Camera", (ox + 12, oy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 60, 0), 1)


def _draw_stencil_map(
    footprints: dict[str, np.ndarray],
    intr: cal.Intrinsics,
    camera_height: float | None = None,
    surface_heights: dict[str, float] | None = None,
) -> np.ndarray:
    canvas = np.full((_MAP_H, _MAP_W, 3), 255, dtype=np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    # Pre-compute smoothed contours once (used in both passes).
    smoothed: list[tuple[str, list]] = []
    for label, coords_2d in footprints.items():
        px = _to_map_px(coords_2d)
        valid = (
            (px[:, 0] >= 0) & (px[:, 0] < _MAP_W) &
            (px[:, 1] >= 0) & (px[:, 1] < _MAP_H)
        )
        if valid.sum() < 3:
            continue
        cell = np.zeros((_MAP_H, _MAP_W), dtype=np.uint8)
        cell[px[valid, 1], px[valid, 0]] = 1
        cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            smoothed.append((label, contours))

    # Pass 1: table shadow (drawn first so outlines appear on top).
    if camera_height is not None and surface_heights is not None:
        for label, contours in smoothed:
            if label not in surface_heights:
                continue
            h_surface = surface_heights[label]
            if camera_height <= h_surface:
                continue
            largest = max(contours, key=cv2.contourArea)
            shadow = _table_shadow_polygon(largest, camera_height, h_surface)
            cv2.fillPoly(canvas, [shadow], (210, 210, 225))

    # Pass 2: footprint outlines and labels.
    for label, contours in smoothed:
        cv2.drawContours(canvas, contours, -1, color=(0, 0, 0), thickness=2)
        largest = max(contours, key=cv2.contourArea)
        lx = int(largest[:, 0, 0].mean())
        ly = int(largest[:, 0, 1].min()) - 8
        cv2.putText(canvas, label.capitalize(), (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    _draw_camera_symbol(canvas, intr)
    return canvas


# ── Main ───────────────────────────────────────────────────────────────────────

def _main():
    print("Loading models...")
    intrinsics = cal.load()
    detector = TagDetector()
    localizer = Localizer(detector, intrinsics.camera_matrix, _cfg.tag.size_m,
                          dist_coeffs=intrinsics.dist_coeffs)
    depth_processor = CalibratedDepthProcessor(localizer)
    segmentor = SemanticSegmentor()

    with Camera() as cam:
        cam.warm_up()

        print("Waiting for 2 AprilTags to calibrate...")
        calib_frame = _auto_calibrate(cam, depth_processor, detector)
        if calib_frame is None:
            print("Cancelled.")
            return

        print(f"Capturing {_NUM_FRAMES} frames for segmentation coverage...")
        frames = [calib_frame]
        for i in range(_NUM_FRAMES - 1):
            time.sleep(_FRAME_INTERVAL_S)
            ok, f = cam.read()
            if ok:
                frames.append(f)
        print(f"  Captured {len(frames)} frames.")

    # --- Depth + segmentation across all frames ---
    # Accumulate per-class 3D points from every frame.
    all_scene_points: list[np.ndarray] = []
    class_points: dict[str, list[np.ndarray]] = {c: [] for c in set(_OBJECT_CLASSES.values())}

    for i, frame in enumerate(frames):
        print(f"Processing frame {i + 1}/{len(frames)}...")

        # Depth — reuse cached result for the calibration frame
        if i == 0:
            depth = depth_processor.last_calibrated_depth
        else:
            depth = depth_processor.process(frame)

        pts = depth_image_to_point_cloud(depth, intrinsics)
        all_scene_points.append(pts)

        seg_results = segmentor.process(frame)
        labels_found = [r.label for r in seg_results]
        print(f"  Segmentation: {labels_found}")

        for raw_label, canonical in _OBJECT_CLASSES.items():
            combined_mask = np.zeros(depth.shape, dtype=bool)
            for seg in seg_results:
                if seg.label.lower() == raw_label:
                    combined_mask |= seg.mask

            if not combined_mask.any():
                continue

            masked_depth = depth.copy()
            masked_depth[~combined_mask] = 0.0
            obj_pts = depth_image_to_point_cloud(masked_depth, intrinsics)
            if len(obj_pts) > 0:
                class_points[canonical].append(obj_pts)

    # --- Floor plane from all accumulated scene points ---
    all_pts = np.vstack(all_scene_points)
    print(f"Total scene points across all frames: {len(all_pts)}")

    print("Running RANSAC to find floor plane...")
    floor_planes = detect_planes(all_pts, n_planes=1, n_iterations=500,
                                 distance_threshold=0.03)
    if not floor_planes:
        print("Floor plane not found.")
        return

    n_floor = floor_planes[0].normal
    d_floor = floor_planes[0].distance
    if d_floor > 0:   # ensure normal points toward camera
        n_floor, d_floor = -n_floor, -d_floor

    u, v = _floor_basis(n_floor)
    print(f"  Floor normal={n_floor.round(3)}  camera height={abs(d_floor):.3f}m")

    # --- Build footprints: height filter → surface RANSAC → project to floor ---
    footprints: dict[str, np.ndarray] = {}
    for canonical, point_lists in class_points.items():
        if not point_lists:
            print(f"  '{canonical}' not detected in any frame.")
            continue
        merged = np.vstack(point_lists)
        above = _filter_above_floor(merged, n_floor, d_floor)
        filtered = merged[above]
        print(f"  '{canonical}': {len(filtered)}/{len(merged)} points above floor.")
        if len(filtered) < 10:
            print(f"    Too few points above floor, skipping.")
            continue
        obj_planes = detect_planes(filtered, n_planes=1, n_iterations=300,
                                   distance_threshold=0.06)
        if obj_planes:
            tilt = float(np.degrees(np.arccos(
                np.clip(abs(float(np.dot(obj_planes[0].normal, n_floor))), 0.0, 1.0)
            )))
            if tilt > _MAX_SURFACE_TILT_DEG:
                print(f"    Surface too tilted ({tilt:.1f}°), skipping RANSAC.")
                surface_pts = filtered
            else:
                surface_pts = filtered[obj_planes[0].inliers]
                print(f"    Surface RANSAC: {len(surface_pts)}/{len(filtered)} inliers, tilt={tilt:.1f}°.")
        else:
            surface_pts = filtered
            print(f"    No surface plane found, using all height-filtered points.")
        component_mask = _largest_floor_component(surface_pts, u, v)
        surface_pts = surface_pts[component_mask]
        print(f"    Largest component: {len(surface_pts)} points.")
        footprints[canonical] = _to_floor_2d(surface_pts, u, v)

    if not footprints:
        print("No target objects found.")

    print("Drawing stencil map...")
    stencil = _draw_stencil_map(footprints, intrinsics)

    _OUTPUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(_OUTPUT_DIR / "stencil_map.png"), stencil)
    cv2.imwrite(str(_OUTPUT_DIR / "frame.jpg"), frames[0])
    print(f"Saved to {_OUTPUT_DIR}/")

    cv2.imshow("Stencil Map", stencil)
    cv2.imshow("Frame", cv2.resize(frames[0], (_cfg.camera.preview_width, _cfg.camera.preview_height)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _main()
