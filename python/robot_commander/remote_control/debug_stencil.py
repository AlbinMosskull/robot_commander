"""
Debug script for the stencil map pipeline.

Instead of hand-drawn ROIs, object masks are produced automatically:
  1. DETR detects objects and returns bounding boxes.
  2. Overlapping boxes of the same class are merged.
  3. SAM refines each merged box into a precise mask.
  4. Masks are filtered to the target classes and used downstream.

Outputs to output/debug/:
  01_frame.jpg          — raw captured frame
  02_depth.png          — depth colormap
  03_depth_overlay.jpg  — depth colormap blended over RGB
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
from robot_commander.image_processing.camera import Camera
from robot_commander.config import load as load_config
from robot_commander.image_processing.tag_detector import TagDetector
from robot_commander.depth_processing.calibrated_depth_processor import CalibratedDepthProcessor
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud
from robot_commander.depth_processing.ransac import detect_planes
from robot_commander.localization.localizer import Localizer
from robot_commander.remote_control.main import (
    _auto_calibrate, _floor_basis, _to_floor_2d, _to_map_px, _filter_above_floor,
    _largest_floor_component, _draw_stencil_map, _OBJECT_CLASSES,
    _NUM_FRAMES, _FRAME_INTERVAL_S,
    _MIN_OBJECT_HEIGHT, _MAX_OBJECT_HEIGHT, _MAX_SURFACE_TILT_DEG,
)
from robot_commander.semantic_understanding.object_detection import ObjectDetector
from robot_commander.semantic_understanding.sam_segmentor import SamSegmentor
from robot_commander.semantic_understanding.types import SegmentationResult

_cfg = load_config()
_DEBUG_DIR = Path("output/debug")

_CLASS_COLORS_BGR = {
    "dining table": (0,  80, 220),  # red-ish
    "couch":        (0, 180,  60),  # green-ish
    # "chair":        (60, 0,  60),   # purple-ish
}

_TARGET_LABELS = set(_OBJECT_CLASSES.values())


# ── Model-based mask detection ─────────────────────────────────────────────────

def _merge_overlapping_boxes(
    detections: list[SegmentationResult],
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Merge overlapping same-class bounding boxes into unified boxes."""
    by_label: dict[str, list[list]] = {}
    for det in detections:
        ys, xs = np.where(det.mask)
        if len(xs) == 0:
            continue
        box = [det.score, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        by_label.setdefault(det.label, []).append(box)

    prompts: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for label, boxes in by_label.items():
        changed = True
        while changed:
            changed = False
            out: list[list] = []
            used = [False] * len(boxes)
            for i in range(len(boxes)):
                if used[i]:
                    continue
                s, ax1, ay1, ax2, ay2 = boxes[i]
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    sj, bx1, by1, bx2, by2 = boxes[j]
                    if ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2:
                        ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                        ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                        s = max(s, sj)
                        used[j] = True
                        changed = True
                out.append([s, ax1, ay1, ax2, ay2])
            boxes = out

        for s, x1, y1, x2, y2 in boxes:
            prompts.append((label, s, (x1, y1, x2, y2)))

    return prompts


def _detect_object_masks(
    frame: np.ndarray,
    detector: ObjectDetector,
    sam: SamSegmentor,
) -> dict[str, np.ndarray]:
    """
    Run DETR → merge → SAM and return one boolean mask per target class.

    If multiple SAM results exist for the same class (shouldn't happen after
    merging, but guards against it), their masks are unioned.
    """
    detections = detector.process(frame)
    detections = [d for d in detections if d.label in _TARGET_LABELS]
    prompts = _merge_overlapping_boxes(detections)
    prompts = [(lbl, sc, box) for lbl, sc, box in prompts if lbl in _TARGET_LABELS]
    sam_results = sam.process(frame, prompts)

    masks: dict[str, np.ndarray] = {}
    for res in sam_results:
        if res.label in masks:
            masks[res.label] |= res.mask
        else:
            masks[res.label] = res.mask.copy()

    return masks


def _resize_mask_to(mask: np.ndarray, shape: tuple) -> np.ndarray:
    """Resize a boolean mask to *shape* (H, W) using nearest-neighbour."""
    h, w = shape[:2]
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


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

def _class_surface_overlay(
    frame: np.ndarray,
    depth: np.ndarray,
    obj_mask: np.ndarray,
    intrinsics: cal.Intrinsics,
    n_floor: np.ndarray, d_floor: float,
    u_vec: np.ndarray, v_vec: np.ndarray,
    label: str,
    path: Path,
) -> None:
    """Height-filter then RANSAC on object points; visualise each stage.

    Green  = RANSAC surface inliers (tabletop / couch cushion)
    Yellow = height-filtered but RANSAC outliers
    Purple = removed by connected-component filter
    Orange = removed by height filter
    Dark   = non-object background
    """
    masked_depth = depth.copy()
    masked_depth[~obj_mask] = 0.0
    obj_pts = depth_image_to_point_cloud(masked_depth, intrinsics)

    if len(obj_pts) < 10:
        print(f"  [{label}] too few points ({len(obj_pts)})")
        return

    heights = obj_pts @ n_floor - d_floor
    above = _filter_above_floor(obj_pts, n_floor, d_floor)
    print(f"  [{label}] height filter: {above.sum()} kept / {(~above).sum()} removed  "
          f"(h range {heights.min():.2f}–{heights.max():.2f} m)")

    filtered = obj_pts[above]
    if len(filtered) >= 10:
        planes = detect_planes(filtered, n_planes=1, n_iterations=300,
                               distance_threshold=0.06)
        if planes:
            tilt = float(np.degrees(np.arccos(
                np.clip(abs(float(np.dot(planes[0].normal, n_floor))), 0.0, 1.0)
            )))
            if tilt > _MAX_SURFACE_TILT_DEG:
                surface_inliers_local = np.ones(len(filtered), dtype=bool)
                print(f"  [{label}] RANSAC plane too tilted ({tilt:.1f}°), rejected")
            else:
                surface_inliers_local = planes[0].inliers
                print(f"  [{label}] surface RANSAC: {surface_inliers_local.sum()}/{len(filtered)} inliers, tilt={tilt:.1f}°")
        else:
            surface_inliers_local = np.ones(len(filtered), dtype=bool)
            print(f"  [{label}] RANSAC found no plane")
    else:
        surface_inliers_local = np.ones(len(filtered), dtype=bool)
        print(f"  [{label}] too few height-filtered points for RANSAC")

    surface_mask = np.zeros(len(obj_pts), dtype=bool)
    above_indices = np.where(above)[0]
    surface_mask[above_indices[surface_inliers_local]] = True
    ransac_outlier_mask = above & ~surface_mask

    surface_pts = obj_pts[surface_mask]
    if len(surface_pts) >= 3:
        comp_keep = _largest_floor_component(surface_pts, u_vec, v_vec)
        kept_global = np.where(surface_mask)[0][comp_keep]
        component_mask = np.zeros(len(obj_pts), dtype=bool)
        component_mask[kept_global] = True
        removed_by_component = surface_mask & ~component_mask
        print(f"  [{label}] largest component: {comp_keep.sum()}/{len(surface_pts)} points kept")
    else:
        component_mask = surface_mask
        removed_by_component = np.zeros(len(obj_pts), dtype=bool)

    u_obj, v_obj = _pixel_coords_from_depth(masked_depth)
    fh, fw = frame.shape[:2]
    dh, dw = depth.shape
    vis = (frame * 0.25).astype(np.uint8)

    def _px(u, v):
        return (u * fw / dw).astype(np.int32).clip(0, fw - 1), \
               (v * fh / dh).astype(np.int32).clip(0, fh - 1)

    u_rm, v_rm = _px(u_obj[~above], v_obj[~above])
    vis[v_rm, u_rm] = (0, 100, 220)
    u_yo, v_yo = _px(u_obj[ransac_outlier_mask], v_obj[ransac_outlier_mask])
    vis[v_yo, u_yo] = (0, 220, 220)
    u_cp, v_cp = _px(u_obj[removed_by_component], v_obj[removed_by_component])
    vis[v_cp, u_cp] = (200, 0, 200)
    u_in, v_in = _px(u_obj[component_mask], v_obj[component_mask])
    vis[v_in, u_in] = (0, 220, 0)

    cv2.putText(vis, f"{label}: green=final  purple=other component  yellow=RANSAC outlier  orange=floor filtered",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), vis)


# ── Main ───────────────────────────────────────────────────────────────────────

def _main():
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    intrinsics = cal.load()
    detector_model = TagDetector()
    localizer = Localizer(detector_model, intrinsics.camera_matrix, _cfg.tag.size_m,
                          dist_coeffs=intrinsics.dist_coeffs)
    depth_processor = CalibratedDepthProcessor(localizer)

    print("Loading DETR detection model...")
    detr = ObjectDetector()
    print("Loading SAM model...")
    sam = SamSegmentor()

    with Camera() as cam:
        cam.warm_up()
        calib_frame = _auto_calibrate(cam, depth_processor, detector_model)
        if calib_frame is None:
            print("Cancelled.")
            return

        frames = [calib_frame]
        for _ in range(_NUM_FRAMES - 1):
            time.sleep(_FRAME_INTERVAL_S)
            ok, f = cam.read()
            if ok:
                frames.append(f)

    # ── 01: raw frame ─────────────────────────────────────────────────────────
    cv2.imwrite(str(_DEBUG_DIR / "01_frame.jpg"), frames[0])

    # ── 02 / 03: depth ────────────────────────────────────────────────────────
    depth0 = depth_processor.last_calibrated_depth
    assert depth0 is not None

    print(f"\n[SHAPE CHECK]")
    print(f"  frame : {frames[0].shape[:2]}  depth : {depth0.shape}", end="  ")
    if frames[0].shape[:2] != depth0.shape:
        print("*** MISMATCH — intrinsics do not match depth pixels ***")
    else:
        print("✓ match")

    depth_vis = _save_depth_vis(depth0, _DEBUG_DIR / "02_depth.png")
    cv2.imwrite(str(_DEBUG_DIR / "03_depth_overlay.jpg"), cv2.addWeighted(
        cv2.resize(frames[0], (depth0.shape[1], depth0.shape[0])), 0.5,
        depth_vis, 0.5, 0,
    ))

    # ── 04: model-predicted masks ─────────────────────────────────────────────
    print("\nRunning DETR + SAM to detect object masks...")
    frame_masks = _detect_object_masks(frames[0], detr, sam)
    print(f"  Detected classes: {list(frame_masks.keys())}")
    _save_mask_vis(frames[0], frame_masks, _DEBUG_DIR / "04_roi_masks.jpg")

    target_masks: dict[str, np.ndarray] = {
        label: _resize_mask_to(mask, depth0.shape)
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
    u_vec, v_vec = _floor_basis(n_floor)
    np.savez(_DEBUG_DIR / "floor_basis.npz", u_vec=u_vec, v_vec=v_vec)

    print(f"\n[PER-CLASS SURFACE]")
    for canonical, obj_mask in target_masks.items():
        fname = "06_table_surface.jpg" if "table" in canonical else "07_couch_surface.jpg"
        _class_surface_overlay(
            frames[0], depth0, obj_mask, intrinsics,
            n_floor, d_floor, u_vec, v_vec, canonical, _DEBUG_DIR / fname,
        )

    # ── 08: floor-projected scatter (all frames) ──────────────────────────────
    depths = [depth0]
    for frame in frames[1:]:
        depths.append(depth_processor.process(frame))

    camera_height = float(abs(d_floor))
    surface_heights: dict[str, float] = {}

    class_2d: dict[str, np.ndarray] = {}
    for canonical, frame_mask in frame_masks.items():
        pts_list = []
        for depth in depths:
            mask = _resize_mask_to(frame_mask, depth.shape)
            md = depth.copy()
            md[~mask] = 0.0
            pts = depth_image_to_point_cloud(md, intrinsics)
            if len(pts):
                pts_list.append(pts)

        if not pts_list:
            print(f"\n[{canonical.upper()}] no depth points in mask")
            continue

        merged = np.vstack(pts_list)
        above = _filter_above_floor(merged, n_floor, d_floor)
        filtered = merged[above]
        print(f"\n[{canonical.upper()}] {len(filtered)}/{len(merged)} points above floor "
              f"across {len(pts_list)} frames")
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
                print(f"  Surface RANSAC: {len(filtered)} inliers, tilt={tilt:.1f}°")
            else:
                print(f"  RANSAC plane too tilted ({tilt:.1f}°), skipped")

        comp_mask = _largest_floor_component(filtered, u_vec, v_vec)
        filtered = filtered[comp_mask]
        print(f"  Largest component: {len(filtered)} points")
        if len(filtered) >= 3:
            class_2d[canonical] = _to_floor_2d(filtered, u_vec, v_vec)
            surface_heights[canonical] = float(np.mean(filtered @ n_floor - d_floor))
            print(f"  Surface height above floor: {surface_heights[canonical]:.3f} m")

    _save_scatter(class_2d, _DEBUG_DIR / "08_scatter.png")

    print(f"\n[SHADOW] camera height: {camera_height:.3f} m  |  surface heights: "
          + ", ".join(f"{k}: {v:.3f} m" for k, v in surface_heights.items()))

    # ── 09: stencil map ───────────────────────────────────────────────────────
    stencil = _draw_stencil_map(class_2d, intrinsics, camera_height, surface_heights)

    # Overlay every detected AprilTag from all captured frames.
    # All instances are drawn individually — duplicates (same ID, different
    # physical copy or detection in a different frame) are each plotted.
    print("\n[APRIL TAGS ON MAP]")
    for frame in frames:
        for tag, (tx, ty, tz) in localizer.localize_all(frame):
            pos_2d = _to_floor_2d(np.array([[tx, ty, tz]]), u_vec, v_vec)
            px = _to_map_px(pos_2d)[0]
            pt = (int(px[0]), int(px[1]))
            # cv2.circle(stencil, pt, 9, (0, 0, 200), -1)
            # cv2.circle(stencil, pt, 9, (0, 0, 0), 1)
            # cv2.putText(stencil, f"Tag {tag.tag_id}", (pt[0] + 12, pt[1] + 4),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 160), 1)
            print(f"  Tag {tag.tag_id}: floor ({pos_2d[0, 0]:.2f}, {pos_2d[0, 1]:.2f}) m")

    cv2.imwrite(str(_DEBUG_DIR / "09_stencil_map.png"), stencil)

    print(f"\nAll debug images saved to {_DEBUG_DIR}/")


if __name__ == "__main__":
    _main()
