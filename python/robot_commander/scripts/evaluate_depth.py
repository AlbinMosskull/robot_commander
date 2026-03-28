"""
Interactive depth evaluation script

1. First a camera should be initialized and warmed up
2. Then, a frame should be captured and the user should click 4 points to define a region of interest
3. Depth should then be captured within this region. (initially using depth anything)
4. It should then be converted to a point cloud
5. Then RANSAC should produce a plane fit to this point cloud
6. Finally, the length and the width of the plane should be printed, and std and max outlier from the plane.

"""

import json
from pathlib import Path

import cv2
import numpy as np

from robot_commander.camera import intrinsics as cal
from robot_commander.camera.camera import Camera
from robot_commander.camera.tag_detector import TagDetector, draw_tags
from robot_commander.depth_processing.calibrated_depth_processor import CalibratedDepthProcessor
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud
from robot_commander.depth_processing.ransac import detect_planes
from robot_commander.localization.localizer import Localizer

_ROI_CACHE_PATH = Path(__file__).parent / ".roi_cache.json"

_TAG_SIZE = 0.03  # physical side length of the AprilTags in metres


def _load_intrinsics() -> tuple[float, float, float, float]:
    intr = cal.load()
    K = intr.camera_matrix
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _load_cached_roi() -> np.ndarray | None:
    """Returns (4, 2) int32 array of points, or None."""
    if not _ROI_CACHE_PATH.exists():
        return None
    try:
        d = json.loads(_ROI_CACHE_PATH.read_text())
        pts = np.array(d["points"], dtype=np.int32)
        if pts.shape == (4, 2):
            return pts
        return None
    except Exception:
        return None


def _save_roi(points: np.ndarray) -> None:
    _ROI_CACHE_PATH.write_text(json.dumps({"points": points.tolist()}))


def _draw_roi_overlay(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    vis = img.copy()
    cv2.polylines(vis, [points.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
    for pt in points:
        cv2.circle(vis, tuple(pt), 6, (0, 255, 0), -1)
    return vis


def _ask_keep_roi(frame: np.ndarray, points: np.ndarray) -> bool:
    """Show the cached ROI polygon. Press Y/Enter to keep, N/Escape to redraw."""
    vis = _draw_roi_overlay(frame, points)
    cv2.putText(vis, "Keep this ROI?  Y = yes   N = redraw",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Cached ROI", vis)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('y'), ord('Y'), 13):   # Y or Enter
            cv2.destroyWindow("Cached ROI")
            return True
        if key in (ord('n'), ord('N'), 27):   # N or Escape
            cv2.destroyWindow("Cached ROI")
            return False


def select_four_points(frame: np.ndarray) -> np.ndarray | None:
    """Show the frame and let the user click 4 points to define a ROI polygon.

    Left-click to place each point in order. The polygon closes automatically
    after the 4th click. Press Escape at any time to cancel.

    Returns:
        (4, 2) int32 array of (x, y) points, or None if cancelled.
    """
    vis = frame.copy()
    points: list[tuple[int, int]] = []
    window_name = "Click 4 points to define ROI  (Escape = cancel)"
    done = [False]

    def mouse_callback(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
            return

        points.append((x, y))
        n = len(points)

        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(vis, str(n), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if n > 1:
            cv2.line(vis, points[-2], points[-1], (0, 255, 0), 2)
        if n == 4:
            cv2.line(vis, points[-1], points[0], (0, 255, 0), 2)
            cv2.putText(vis, "Done — press any key to continue",
                        (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            done[0] = True

        cv2.imshow(window_name, vis)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        display = vis.copy()
        remaining = 4 - len(points)
        if remaining > 0:
            label = f"Click point {len(points) + 1} of 4"
            cv2.putText(display, label, (10, 30), font, 0.8, (0, 255, 255), 2)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:   # Escape — cancel
            cv2.destroyWindow(window_name)
            return None
        if done[0]:
            cv2.waitKey(400)   # brief pause so user sees the closed polygon
            break

    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.int32)


def _roi_mask_and_bbox(
    points: np.ndarray, frame_shape: tuple
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return a boolean mask (H, W) for the polygon and its axis-aligned bounding box."""
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)

    full_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(full_mask, [points.reshape((-1, 1, 2))], 1)
    mask_crop = full_mask[y1:y2, x1:x2].astype(bool)
    return mask_crop, (int(x1), int(y1), int(x2), int(y2))


def plane_dimensions(points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Compute length and width of inlier points projected onto the plane.

    Returns:
        (length, width) where length >= width.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    in_plane = centered - (centered @ normal)[:, None] * normal
    _, _, vh = np.linalg.svd(in_plane, full_matrices=False)
    coords = in_plane @ vh[:2].T
    extents = coords.max(axis=0) - coords.min(axis=0)
    return float(extents.max()), float(extents.min())


def _show_results(
    frame: np.ndarray,
    depth_crop: np.ndarray,
    roi_points: np.ndarray,
    mask_crop: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox

    # Left panel: RGB crop with polygon overlaid
    rgb_crop = frame[y1:y2, x1:x2].copy()
    local_pts = roi_points - np.array([x1, y1], dtype=np.int32)
    cv2.polylines(rgb_crop, [local_pts.reshape((-1, 1, 2))],
                  isClosed=True, color=(0, 255, 0), thickness=2)

    # Right panel: depth as a colourmap, masked to ROI
    depth_display = depth_crop.copy()
    depth_display[~mask_crop] = 0.0
    valid_vals = depth_display[mask_crop]
    if valid_vals.size:
        d_min, d_max = valid_vals.min(), valid_vals.max()
        norm = np.zeros_like(depth_display, dtype=np.uint8)
        if d_max > d_min:
            norm[mask_crop] = (
                (depth_display[mask_crop] - d_min) / (d_max - d_min) * 255
            ).astype(np.uint8)
        depth_color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        depth_color[~mask_crop] = 0
    else:
        depth_color = np.zeros((*depth_crop.shape, 3), dtype=np.uint8)

    # Resize both panels to the same height before hstacking
    h = max(rgb_crop.shape[0], depth_color.shape[0])
    def _pad_to_height(img, target_h):
        pad = target_h - img.shape[0]
        if pad > 0:
            img = np.pad(img, ((0, pad), (0, 0), (0, 0)))
        return img

    panel = np.hstack([_pad_to_height(rgb_crop, h), _pad_to_height(depth_color, h)])
    cv2.putText(panel, "Selected region", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(panel, "Depth map", (rgb_crop.shape[1] + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Results — press any key to exit", panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _capture_calibration_frame(
    cam: Camera, processor: CalibratedDepthProcessor, detector: TagDetector
) -> np.ndarray | None:
    """Show live feed until the user presses C to calibrate and capture, or Q to cancel."""
    window = "Show 2 AprilTags then press C to calibrate — Q to quit"
    while True:
        ok, frame = cam.read()
        if not ok:
            cv2.destroyWindow(window)
            return None

        tags = detector.detect(frame)
        vis = draw_tags(frame, tags)
        n = len(tags)
        color = (0, 255, 0) if n >= 2 else (0, 255, 255)
        cv2.putText(vis, f"{n}/2 tags visible — press C to calibrate",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow(window, cv2.resize(vis, (960, 540)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window)
            return None
        if key in (ord('c'), ord('C')):
            if processor.calibrate(frame):
                cv2.destroyWindow(window)
                return frame
            # calibrate() already printed the failure reason; keep looping


def main():
    print("Loading depth model...")
    intrinsics = cal.load()
    detector = TagDetector()
    localizer = Localizer(detector, intrinsics.camera_matrix, _TAG_SIZE,
                          dist_coeffs=intrinsics.dist_coeffs)
    processor = CalibratedDepthProcessor(localizer)

    print("Opening camera...")
    with Camera(0) as cam:
        print("Warming up camera...")
        cam.warm_up()

        frame = _capture_calibration_frame(cam, processor, detector)
        if frame is None:
            print("Cancelled.")
            return

    cached = _load_cached_roi()
    if cached is not None and _ask_keep_roi(frame, cached):
        roi_points = cached
    else:
        while True:
            print("Click 4 points on the frame to define the region of interest.")
            roi_points = select_four_points(frame)
            if roi_points is None:
                print("Cancelled.")
                return
            if _ask_keep_roi(frame, roi_points):
                break
        _save_roi(roi_points)

    mask_crop, (x1, y1, x2, y2) = _roi_mask_and_bbox(roi_points, frame.shape)
    print(f"Selected ROI bounding box: ({x1}, {y1}) → ({x2}, {y2})")

    fx, fy, cx, cy = _load_intrinsics()

    print("Running depth estimation...")
    depth_full = processor.process(frame)
    depth_crop = depth_full[y1:y2, x1:x2].copy()

    # Zero out pixels outside the polygon so they are excluded from point cloud
    depth_crop[~mask_crop] = 0.0

    cx_crop = cx - x1
    cy_crop = cy - y1

    print("Building point cloud...")
    points = depth_image_to_point_cloud(depth_crop, fx, fy, cx_crop, cy_crop)

    if len(points) < 10:
        print("Not enough valid depth points in the selected region.")
        return

    print(f"Point cloud has {len(points)} points. Running RANSAC...")
    planes = detect_planes(points, n_planes=1, n_iterations=200, distance_threshold=0.02)

    if not planes:
        print("No plane detected in the selected region.")
        return

    plane = planes[0]
    inlier_points = points[plane.inliers]
    distances = np.abs(inlier_points @ plane.normal - plane.distance)

    length, width = plane_dimensions(inlier_points, plane.normal)

    print(f"\n--- Plane fit results ---")
    print(f"Inliers:       {plane.inliers.sum()} / {len(points)}")
    print(f"Length:        {length:.3f} m")
    print(f"Width:         {width:.3f} m")
    print(f"Std dev:       {distances.std():.4f} m")
    print(f"Max outlier:   {distances.max():.4f} m")


if __name__ == "__main__":
    main()
