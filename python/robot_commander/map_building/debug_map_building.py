"""
Visualization helpers for the stencil map building pipeline.
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_commander.image_processing import intrinsics as cal
from robot_commander.depth_processing.ransac import Plane
from robot_commander.localization.localizer import Localizer


def pixel_coords_from_depth(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dh, dw = depth.shape
    uu, vv = np.meshgrid(np.arange(dw), np.arange(dh))
    valid = depth > 0
    return uu[valid], vv[valid]


def ransac_overlay(
    frame: np.ndarray,
    depth: np.ndarray,
    inlier_mask: np.ndarray,
    color_inlier: tuple,
    color_outlier: tuple = (40, 40, 40),
) -> np.ndarray:
    fh, fw = frame.shape[:2]
    dh, dw = depth.shape
    u_px, v_px = pixel_coords_from_depth(depth)
    u_f = (u_px * fw / dw).astype(np.int32).clip(0, fw - 1)
    v_f = (v_px * fh / dh).astype(np.int32).clip(0, fh - 1)
    vis = frame.copy()
    vis[v_f[~inlier_mask], u_f[~inlier_mask]] = color_outlier
    vis[v_f[inlier_mask],  u_f[inlier_mask]]  = color_inlier
    return vis


def check_depth_and_save_vis(color: np.ndarray, depth: np.ndarray, path: Path) -> None:
    print(f"\n[SHAPE CHECK]")
    print(f"  frame : {color.shape[:2]}  depth : {depth.shape}", end="  ")
    if color.shape[:2] != depth.shape:
        print("*** MISMATCH — intrinsics do not match depth pixels ***")
    else:
        print("✓ match")
    
    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), colored)


def save_mask_vis(
    frame: np.ndarray,
    masks: dict[str, np.ndarray],
    class_colors_bgr: dict[str, tuple],
    path: Path,
) -> None:
    vis = frame.copy()
    for label, mask in masks.items():
        color = class_colors_bgr.get(label, (200, 200, 200))
        vis[mask] = (vis[mask] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(vis, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        np.array(color, dtype=np.uint8).tolist(), 2)
    cv2.imwrite(str(path), vis)


def save_scatter(class_2d: dict[str, np.ndarray], path: Path) -> None:
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


def check_tag_normals(
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


def save_floor_vis(
    frame: np.ndarray,
    depth: np.ndarray,
    floor: Plane,
    path: Path,
) -> None:
    """Print floor plane diagnostics and save the RANSAC inlier overlay."""
    print(f"\n[POINT CLOUD] {floor.inliers.size} points (frame 0)")
    print(f"\n[FLOOR PLANE]")
    print(f"  normal        : {floor.normal.round(4)}")
    print(f"  camera height : {abs(floor.distance):.3f} m")
    tilt = np.degrees(np.arccos(np.clip(abs(floor.normal[1]), 0, 1)))
    print(f"  tilt from cam-Y axis: {tilt:.1f}°")
    print(f"  inliers       : {floor.inliers.sum()} / {floor.inliers.size}")

    overlay = ransac_overlay(frame, depth, floor.inliers,
                             color_inlier=(0, 220, 0), color_outlier=(40, 40, 40))
    cv2.imwrite(str(path), overlay)


def save_surface_vis(
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
