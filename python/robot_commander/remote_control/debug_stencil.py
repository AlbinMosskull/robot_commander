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
import numpy as np

from robot_commander.image_processing import intrinsics as cal
from robot_commander.image_processing.camera import FromFileCamera
from robot_commander.config import load as load_config
from robot_commander.image_processing.tag_detector import TagDetector
from robot_commander.depth_processing.calibrated_depth_processor import CalibratedDepthProcessor
from robot_commander.localization.localizer import Localizer
from robot_commander.semantic_understanding.detection_segmentor import DetectionSegmentor
from robot_commander.remote_control.map_drawing import draw_stencil_map
from robot_commander.remote_control.map_geometry import FootprintResult, to_floor_2d, build_footprints
from robot_commander.remote_control.debug_map_building import (
    save_depth_vis,
    save_mask_vis,
    save_scatter,
    check_tag_normals,
    save_surface_vis,
    debug_floor_plane,
)

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


def _main():
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    intrinsics = cal.load()
    detector_model = TagDetector()
    localizer = Localizer(detector_model, intrinsics.camera_matrix, _cfg.tag.size_m,
                          dist_coeffs=intrinsics.dist_coeffs)
    depth_processor = CalibratedDepthProcessor()
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

    # ── 02: depth ─────────────────────────────────────────────────────────────
    print(f"\n[SHAPE CHECK]")
    print(f"  frame : {frames[0].shape[:2]}  depth : {depth0.shape}", end="  ")
    if frames[0].shape[:2] != depth0.shape:
        print("*** MISMATCH — intrinsics do not match depth pixels ***")
    else:
        print("✓ match")

    save_depth_vis(depth0, _DEBUG_DIR / "02_depth.png")

    # ── 04: model-predicted masks ─────────────────────────────────────────────
    print("\nRunning DETR + SAM to detect object masks...")
    frame_masks = {
        r.label: r.mask
        for r in segmentor.process(frames[0])
        if r.label in _TARGET_LABELS
    }
    print(f"  Detected classes: {list(frame_masks.keys())}")
    save_mask_vis(frames[0], frame_masks, _CLASS_COLORS_BGR, _DEBUG_DIR / "04_roi_masks.jpg")

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
    floor = debug_floor_plane(frames[0], depth0, intrinsics, _DEBUG_DIR / "05_floor_ransac.jpg")
    if floor is None:
        return
    n_floor, d_floor = floor

    print(f"\n[APRIL TAG NORMALS vs FLOOR]")
    check_tag_normals(localizer, frames[0], n_floor)

    # ── 06 / 07: per-class surface overlay ────────────────────────────────────
    camera_height = float(abs(d_floor))
    depths = [depth0] + [depth_processor.process(f) for f in frames[1:]]
    result: FootprintResult = build_footprints(depths, frame_masks, intrinsics, n_floor, d_floor)
    np.savez(_DEBUG_DIR / "floor_basis.npz", u_vec=result.u_floor, v_vec=result.v_floor)

    print(f"\n[PER-CLASS SURFACE]")
    for canonical, pts_3d in result.label_points.items():
        fname = "06_table_surface.jpg" if "table" in canonical else "07_couch_surface.jpg"
        save_surface_vis(frames[0], pts_3d, intrinsics, canonical, _DEBUG_DIR / fname)

    # ── 08: floor-projected scatter (all frames) ──────────────────────────────
    save_scatter(result.footprints, _DEBUG_DIR / "08_scatter.png")

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