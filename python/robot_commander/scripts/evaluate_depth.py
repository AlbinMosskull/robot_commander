"""
Interactive depth evaluation script

1. First a camera should be initialized and warmed up
2. Then, a frame should be captured and the user should be allowed to draw a bounding box on it
3. Depth should then be captured within this region. (initially using depth anything)
4. It should then be converted to a point cloud
5. Then RANSAC should produce a plane fit to this point cloud
6. Finally, the length and the width of the plane should be printed, and std and max outlier from the plane.

"""

import cv2
import numpy as np

from robot_commander.camera.camera import Camera
from robot_commander.depth_processing.depth_processor import DepthProcessor
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud
from robot_commander.depth_processing.ransac import detect_planes

# Approximate intrinsics for a typical webcam at 640x360.
# Replace with calibrated values for better accuracy.
FX = 500.0
FY = 500.0


def draw_bounding_box(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Show the frame and let the user drag a bounding box with the mouse.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates, or None if the user cancelled.
    """
    box: list[tuple[int, int]] = []
    drawing = False
    current: list[tuple[int, int]] = [(-1, -1)]
    done = False

    clone = frame.copy()
    window = "Draw bounding box — drag to select, Enter to confirm, Esc to cancel"

    def on_mouse(event, x, y, *_):
        nonlocal drawing, done
        if event == cv2.EVENT_LBUTTONDOWN:
            box.clear()
            box.append((x, y))
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            box.append((x, y))
            drawing = False

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        img = clone.copy()
        if len(box) >= 1 and current[0] != (-1, -1) and drawing:
            cv2.rectangle(img, box[0], current[0], (0, 255, 0), 2)
        elif len(box) == 2:
            cv2.rectangle(img, box[0], box[1], (0, 255, 0), 2)
        cv2.imshow(window, img)
        key = cv2.waitKey(16) & 0xFF
        if key == 13 and len(box) == 2:  # Enter
            break
        if key == 27:  # Esc
            cv2.destroyWindow(window)
            return None

    cv2.destroyWindow(window)
    x1 = min(box[0][0], box[1][0])
    y1 = min(box[0][1], box[1][1])
    x2 = max(box[0][0], box[1][0])
    y2 = max(box[0][1], box[1][1])
    return x1, y1, x2, y2


def plane_dimensions(points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Compute the length and width of a set of (inlier) points on a plane.

    Projects points onto the plane and returns the extents along the two
    principal axes found by PCA.

    Returns:
        (length, width) where length >= width.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # Remove normal component to get in-plane coordinates
    in_plane = centered - (centered @ normal)[:, None] * normal
    _, _, vh = np.linalg.svd(in_plane, full_matrices=False)
    coords = in_plane @ vh[:2].T  # (N, 2) — projection onto two principal axes
    extents = coords.max(axis=0) - coords.min(axis=0)
    return float(extents.max()), float(extents.min())


def main():
    print("Loading depth model...")
    processor = DepthProcessor("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")

    print("Opening camera...")
    with Camera(0) as cam:
        print("Warming up camera...")
        cam.warm_up()

        ret, frame = cam.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

    print("Draw a bounding box over the region of interest.")
    bbox = draw_bounding_box(frame)
    if bbox is None:
        print("Cancelled.")
        return

    x1, y1, x2, y2 = bbox
    print(f"Selected region: ({x1}, {y1}) → ({x2}, {y2})")

    cropped = frame[y1:y2, x1:x2]
    h, w = frame.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    print("Running depth estimation...")
    depth_full = processor.process(frame)
    depth_crop = depth_full[y1:y2, x1:x2]

    # Adjust principal point for the cropped region.
    cx_crop = cx - x1
    cy_crop = cy - y1

    print("Building point cloud...")
    points = depth_image_to_point_cloud(depth_crop, FX, FY, cx_crop, cy_crop)

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

    # Visualise the selected region with depth overlay
    depth_vis = cv2.normalize(depth_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    combined = np.hstack([cropped, depth_color])
    cv2.imshow("Region | Depth", combined)
    print("\nPress any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
