"""
Debug script for the stencil map pipeline.

Instead of hand-drawn ROIs, object masks are produced automatically:
  1. DETR detects objects and returns bounding boxes.
  2. Overlapping boxes of the same class are merged.
  3. SAM refines each merged box into a precise mask.
  4. Masks are filtered to the target classes and used downstream.

Outputs to plots/debug/:
  01_frame.jpg          — raw captured frame
  02_depth.png          — depth colormap
  04_roi_masks.jpg      — model-predicted masks coloured by class
  05_floor_ransac.jpg   — floor RANSAC inliers (green) vs rest (dark) on frame
  06_table_surface.jpg  — table surface: green=final  purple=other component  yellow=RANSAC outlier  orange=floor filtered
  07_couch_surface.jpg  — couch surface (same colour scheme)
  08_scatter.png        — 2D floor-projected scatter per class (all frames)
  09_stencil_map.png    — final stencil map

Terminal also prints:
  - frame shape vs depth shape
  - AprilTag surface normals vs floor RANSAC normal
  - floor plane stats
  - per-class depth / position ranges
"""

import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_commander.image_processing import intrinsics as cal
from robot_commander.image_processing.camera import FromFileCamera
from robot_commander.config import load as load_config
from robot_commander.image_processing.tag_detector import TagDetector
from robot_commander.depth_processing.calibrated_depth_processor import CalibratedDepthProcessor
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud
from robot_commander.depth_processing.ransac import detect_planes
from robot_commander.localization.localizer import Localizer
from robot_commander.semantic_understanding.detection_segmentor import DetectionSegmentor
from robot_commander.remote_control.map_drawing import draw_stencil_map
from robot_commander.remote_control.map_geometry import FootprintResult, to_floor_2d, build_footprints

_cfg = load_config()
_DEBUG_DIR = Path("plots/debug")

_OBJECT_CLASSES: dict[str, str] = {
    "dining table": "dining table",
    "couch":        "couch",
    # "chair":        "chair",
}
_NUM_FRAMES = 5
_FRAME_INTERVAL_S = 0.3

_CLASS_COLORS_BGR = {
    "dining table": (0,  80, 220),  # red-ish
    "couch":        (0, 180,  60),  # green-ish
    # "chair":        (60, 0,  60),   # purple-ish
}

_TARGET_LABELS = set(_OBJECT_CLASSES.values())


# ── Visualisation helpers ──────────────────────────────────────────────────────

def _pixel_coords_from_depth(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dh, dw = depth.shape
    uu, vv = np.meshgrid(np.arange(dw), np.arange(dh))
    valid = depth > 0
    return uu[valid], vv[valid]


def _ransac_overlay(
    frame: np.ndarray,
    depth: np.ndarray,
    inlier_mask: np.ndarray,
    color_inlier: tuple,
    color_outlier: tuple = (40, 40, 40),
) -> np.ndarray:
    fh, fw = frame.shape[:2]
    dh, dw = depth.shape
    u_px, v_px = _pixel_coords_from_depth(depth)
    u_f = (u_px * fw / dw).astype(np.int32).clip(0, fw - 1)
    v_f = (v_px * fh / dh).astype(np.int32).clip(0, fh - 1)
    vis = frame.copy()
    vis[v_f[~inlier_mask], u_f[~inlier_mask]] = color_outlier
    vis[v_f[inlier_mask],  u_f[inlier_mask]]  = color_inlier
    return vis


def _save_depth_vis(depth: np.ndarray, path: Path) -> np.ndarray:
    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), colored)
    return colored


def _save_mask_vis(
    frame: np.ndarray, masks: dict[str, np.ndarray], path: Path
) -> None:
    """Save frame with model-predicted masks as coloured overlays."""
    vis = frame.copy()
    for label, mask in masks.items():
        color = _CLASS_COLORS_BGR.get(label, (200, 200, 200))
        vis[mask] = (vis[mask] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(vis, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        np.array(color, dtype=np.uint8).tolist(), 2)
    cv2.imwrite(str(path), vis)


def _save_scatter(class_2d: dict[str, np.ndarray], path: Path) -> None:
    colors = {"dining table": "tab:red", "couch": "tab:blue"}
    fig, ax = plt.subplots(figsize=(8, 8))
    for label, pts in class_2d.items():
        if not len(pts):
            continue
        idx = np.random.choice(len(pts), min(5000, len(pts)), replace=False)
        ax.scatter(pts[idx, 0], pts[idx, 1], s=1, alpha=0.3,
                   color=colors.get(label, "gray"), label=label)
    ax.set_xlabel("right (m)")
    ax.set_ylabel("forward (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Floor-projected points — camera at origin")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── AprilTag normal check ──────────────────────────────────────────────────────

def _check_tag_normals(
    localizer: Localizer, frame: np.ndarray, floor_normal: np.ndarray
) -> None:
    tags = localizer._detector.detect(frame)
    if not tags:
        print("  No tags detected in calibration frame.")
        return

    records = []
    for tag in tags:
        ok, rvec, tvec = cv2.solvePnP(
            localizer._obj_points,
            tag.corners.astype(np.float32),
            localizer._camera_matrix,
            localizer._dist_coeffs,
        )
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        tag_normal = R[:, 2]
        if np.dot(tag_normal, floor_normal) < 0:
            tag_normal = -tag_normal
        angle = float(np.degrees(np.arccos(np.clip(np.dot(tag_normal, floor_normal), -1.0, 1.0))))
        z = float(tvec.flatten()[2])
        records.append((z, tag.tag_id, tag_normal, angle))
        print(f"  Tag {tag.tag_id:3d}: z={z:.3f}m  normal={tag_normal.round(3)}"
              f"  angle_vs_floor={angle:.2f}°")

    if not records:
        return
    records.sort(key=lambda r: r[0], reverse=True)
    z, tid, tnorm, angle = records[0]
    print(f"\n  Further tag (ID={tid}, z={z:.3f}m):")
    print(f"    tag normal   : {tnorm.round(4)}")
    print(f"    floor normal : {floor_normal.round(4)}")
    print(f"    angle between: {angle:.2f}°   (0° = tag perfectly flat on floor)")
    if angle > 15:
        print("    *** Large angle — tag may not be lying flat, "
              "or floor RANSAC found the wrong surface ***")


# ── Per-class surface overlay ──────────────────────────────────────────────────

def _save_surface_vis(
    frame: np.ndarray,
    pts_3d: np.ndarray,
    intrinsics: cal.Intrinsics,
    label: str,
    path: Path,
) -> None:
    fh, fw = frame.shape[:2]
    vis = (frame * 0.25).astype(np.uint8)
    u = (intrinsics.fx * pts_3d[:, 0] / pts_3d[:, 2] + intrinsics.cx).astype(np.int32).clip(0, fw - 1)
    v = (intrinsics.fy * pts_3d[:, 1] / pts_3d[:, 2] + intrinsics.cy).astype(np.int32).clip(0, fh - 1)
    vis[v, u] = (0, 220, 0)
    cv2.putText(vis, f"{label}: green=final surface", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), vis)


# ── Main ───────────────────────────────────────────────────────────────────────

def _main():
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    intrinsics = cal.load()
    detector_model = TagDetector()
    localizer = Localizer(detector_model, intrinsics.camera_matrix, _cfg.tag.size_m,
                          dist_coeffs=intrinsics.dist_coeffs)
    depth_processor = CalibratedDepthProcessor()

    print("Loading DETR + SAM models...")
    segmentor = DetectionSegmentor()

    image_path = Path("images/example_input/scene_image.jpg")
    with FromFileCamera(image_path) as cam:
        cam.warm_up()
        _, frame0 = cam.read()
        calib_result = depth_processor.calibrate(frame0, localizer)
        if calib_result is None:
            print("Calibration failed — ensure 2 AprilTags are visible.")
            return
        _, depth0 = calib_result

        frames = [frame0]
        for _ in range(_NUM_FRAMES - 1):
            time.sleep(_FRAME_INTERVAL_S)
            ok, f = cam.read()
            if ok:
                frames.append(f)

    # ── 01: raw frame ─────────────────────────────────────────────────────────
    cv2.imwrite(str(_DEBUG_DIR / "01_frame.jpg"), frames[0])

    # ── 02 / 03: depth ────────────────────────────────────────────────────────

    print(f"\n[SHAPE CHECK]")
    print(f"  frame : {frames[0].shape[:2]}  depth : {depth0.shape}", end="  ")
    if frames[0].shape[:2] != depth0.shape:
        print("*** MISMATCH — intrinsics do not match depth pixels ***")
    else:
        print("✓ match")

    _save_depth_vis(depth0, _DEBUG_DIR / "02_depth.png")


    # ── 04: model-predicted masks ─────────────────────────────────────────────
    print("\nRunning DETR + SAM to detect object masks...")
    frame_masks = {
        r.label: r.mask
        for r in segmentor.process(frames[0])
        if r.label in _TARGET_LABELS
    }
    print(f"  Detected classes: {list(frame_masks.keys())}")
    _save_mask_vis(frames[0], frame_masks, _DEBUG_DIR / "04_roi_masks.jpg")

    target_masks: dict[str, np.ndarray] = {
        label: cv2.resize(mask.astype(np.uint8), (depth0.shape[1], depth0.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        for label, mask in frame_masks.items()
    }

    if len(target_masks) == 2:
        labs = list(target_masks)
        overlap = (target_masks[labs[0]] & target_masks[labs[1]]).sum()
        print(f"  Mask overlap '{labs[0]}' ∩ '{labs[1]}': {overlap} px")

    # ── 05: floor RANSAC ──────────────────────────────────────────────────────
    all_pts0 = depth_image_to_point_cloud(depth0, intrinsics)
    print(f"\n[POINT CLOUD] {len(all_pts0)} points (frame 0)")

    floor_planes = detect_planes(all_pts0, n_planes=1, n_iterations=500,
                                 distance_threshold=0.03)
    if not floor_planes:
        print("Floor plane not found.")
        return

    n_floor = floor_planes[0].normal
    d_floor = floor_planes[0].distance
    if d_floor > 0:
        n_floor, d_floor = -n_floor, -d_floor

    print(f"\n[FLOOR PLANE]")
    print(f"  normal        : {n_floor.round(4)}")
    print(f"  camera height : {abs(d_floor):.3f} m")
    tilt = np.degrees(np.arccos(np.clip(abs(n_floor[1]), 0, 1)))
    print(f"  tilt from cam-Y axis: {tilt:.1f}°")
    print(f"  inliers       : {floor_planes[0].inliers.sum()} / {len(all_pts0)}")

    floor_overlay = _ransac_overlay(
        frames[0], depth0, floor_planes[0].inliers,
        color_inlier=(0, 220, 0), color_outlier=(40, 40, 40),
    )
    cv2.imwrite(str(_DEBUG_DIR / "05_floor_ransac.jpg"), floor_overlay)

    print(f"\n[APRIL TAG NORMALS vs FLOOR]")
    _check_tag_normals(localizer, frames[0], n_floor)

    # ── 06 / 07: per-class surface overlay ────────────────────────────────────
    camera_height = float(abs(d_floor))
    depths = [depth0] + [depth_processor.process(f) for f in frames[1:]]
    result: FootprintResult = build_footprints(depths, frame_masks, intrinsics, n_floor, d_floor)
    np.savez(_DEBUG_DIR / "floor_basis.npz", u_vec=result.u_floor, v_vec=result.v_floor)

    print(f"\n[PER-CLASS SURFACE]")
    for canonical, pts_3d in result.label_points.items():
        fname = "06_table_surface.jpg" if "table" in canonical else "07_couch_surface.jpg"
        _save_surface_vis(frames[0], pts_3d, intrinsics, canonical, _DEBUG_DIR / fname)

    # ── 08: floor-projected scatter (all frames) ──────────────────────────────
    _save_scatter(result.footprints, _DEBUG_DIR / "08_scatter.png")

    print(f"\n[SHADOW] camera height: {camera_height:.3f} m  |  surface heights: "
          + ", ".join(f"{k}: {v:.3f} m" for k, v in result.surface_heights.items()))

    # ── 09: stencil map ───────────────────────────────────────────────────────
    stencil = draw_stencil_map(result.footprints, intrinsics, camera_height, result.surface_heights)

    print("\n[APRIL TAGS ON MAP]")
    for frame in frames:
        for tag, (tx, ty, tz) in localizer.localize_all(frame):
            pos_2d = to_floor_2d(np.array([[tx, ty, tz]]), result.u_floor, result.v_floor)
            print(f"  Tag {tag.tag_id}: floor ({pos_2d[0, 0]:.2f}, {pos_2d[0, 1]:.2f}) m")

    cv2.imwrite(str(_DEBUG_DIR / "09_stencil_map.png"), stencil)

    print(f"\nAll debug images saved to {_DEBUG_DIR}/")


if __name__ == "__main__":
    _main()
