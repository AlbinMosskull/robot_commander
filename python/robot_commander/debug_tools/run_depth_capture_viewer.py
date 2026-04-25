import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from robot_commander.config import load as load_config
from robot_commander.agent.adeept.adeept_transforms import CAMERA_T_SENSOR_CENTER
from robot_commander.depth_processing.cone_depth_processor import ConeDepthProcessor, ConeGeometry
from robot_commander.depth_processing.cone_depth_rays import depth_to_rays
from robot_commander.depth_processing.depth_capture import load, rays_to_ends
from robot_commander.depth_processing.depth_planes import extract_landmark_planes_debug
from robot_commander.depth_processing.ransac import detect_planes
from robot_commander.depth_processing.ultrasonic_plane_validator import PlaneValidationResult
from robot_commander.image_processing.intrinsics import Intrinsics

_DEFAULT_PATH = Path(__file__).parent / "latest_depth_capture.npz"
_HEADING_ARROW_LENGTH_M = 0.3
_PLANE_COLORS = ["red", "lime", "cyan"]
_cfg = load_config()


def _build_depth_processor(capture) -> ConeDepthProcessor:
    cone_geometry = ConeGeometry(half_angle_radians=math.radians(_cfg.depth.cone_half_angle_deg))
    return ConeDepthProcessor(
        intrinsics=capture.intrinsics,
        camera_T_sensor=CAMERA_T_SENSOR_CENTER,
        cone_geometry=cone_geometry,
    )


def _print_validation(validation: PlaneValidationResult) -> None:
    print("\n--- Ultrasonic plane validation ---")
    if not validation.all_candidates:
        print("  No planes detected")
    for i, candidate in enumerate(validation.all_candidates):
        selected = candidate is validation.best_candidate
        tag = " [SELECTED]" if selected else ""
        print(
            f"  Plane {i}: fill={candidate.cone_fill_fraction:.3f}"
            f"  normal_angle={candidate.normal_angle_deg:.1f}°{tag}"
        )
    if validation.disqualification_reason is None:
        print("  Result: VALID")
    else:
        print(f"  Result: DISQUALIFIED — {validation.disqualification_reason}")
    print()


def _show(
    capture_path: Path,
    save_path: Path | None = None,
) -> None:
    capture = load(capture_path)

    depth_processor = _build_depth_processor(capture)
    raw_depth, calibrated_depth, cone_mask, validation_mask, validation = depth_processor.process_with_validation(
        capture.frame, capture.ultrasonic_min
    )

    print(f"Ultrasonic min reading: {capture.ultrasonic_min:.4f} m")
    print(f"agent: ({capture.agent_x:.4f}, {capture.agent_y:.4f})  heading: {capture.heading:.4f} rad")
    depth_label = "calibrated" if capture.is_calibrated else "raw metric"
    print(f"depth ({depth_label}) range: [{calibrated_depth.min():.4f}, {calibrated_depth.max():.4f}] m")
    print(f"depth ({depth_label}, cone only) range: [{calibrated_depth[cone_mask].min():.4f}, {calibrated_depth[cone_mask].max():.4f}] m")
    _print_validation(validation)

    rays = depth_to_rays(calibrated_depth, capture.intrinsics, capture.agent_x, capture.agent_y, capture.heading)
    ray_ends = rays_to_ends(rays)
    print(f"ray_ends (regenerated): {ray_ends.shape}")

    is_valid = validation.disqualification_reason is None
    display_depth = calibrated_depth if capture.is_calibrated else raw_depth
    calibration_tag = "calibrated" if capture.is_calibrated else "uncalibrated (raw metric)"

    landmark_debug = extract_landmark_planes_debug(
        calibrated_depth if is_valid else raw_depth,
        capture.intrinsics, capture.agent_x, capture.agent_y, capture.heading,
    )

    fig, axes = plt.subplots(1, 6, figsize=(33, 5))
    fig.suptitle(f"Depth capture — {capture_path.name}  [{calibration_tag}]")

    _draw_frame_with_cone_mask(axes[0], capture.frame, cone_mask)
    _draw_depth_map(axes[1], display_depth, title=f"Depth ({calibration_tag})")
    _draw_top_down_rays(axes[2], capture.agent_x, capture.agent_y, capture.heading, ray_ends)
    _draw_plane_inliers(axes[3], display_depth, validation_mask, validation)
    _draw_all_planes(axes[4], raw_depth, capture.intrinsics)
    _draw_landmark_plane_debug(axes[5], capture.agent_x, capture.agent_y, capture.heading, landmark_debug)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


def _draw_frame_with_cone_mask(ax, frame: np.ndarray, cone_mask: np.ndarray) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[cone_mask] = (overlay[cone_mask] * 0.5 + np.array([0, 180, 0]) * 0.5).astype(np.uint8)
    ax.imshow(overlay)
    ax.set_title("Frame + cone mask")
    ax.axis("off")


def _draw_depth_map(ax, depth: np.ndarray, title: str = "Depth") -> None:
    im = ax.imshow(depth, cmap="plasma", vmin=0)
    plt.colorbar(im, ax=ax, label="depth (m)")
    ax.set_title(title)
    ax.axis("off")

    def format_coord(x, y):
        col, row = int(x + 0.5), int(y + 0.5)
        if 0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]:
            return f"x={x:.1f}, y={y:.1f}, depth={depth[row, col]:.4f} m"
        return f"x={x:.1f}, y={y:.1f}"

    ax.format_coord = format_coord


def _draw_top_down_rays(ax, agent_x: float, agent_y: float, heading: float, ray_ends: np.ndarray) -> None:
    ax.set_aspect("equal")
    ax.set_title("Top-down rays (world frame, m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.plot(agent_x, agent_y, "bo", markersize=8, label="agent")

    dx = _HEADING_ARROW_LENGTH_M * math.cos(heading)
    dy = _HEADING_ARROW_LENGTH_M * math.sin(heading)
    ax.annotate(
        "",
        xy=(agent_x + dx, agent_y + dy),
        xytext=(agent_x, agent_y),
        arrowprops={"arrowstyle": "->", "color": "blue", "lw": 1.5},
    )

    if len(ray_ends) > 0:
        for end_x, end_y in ray_ends:
            ax.plot(
                [agent_x, end_x],
                [agent_y, end_y],
                color="orange",
                linewidth=1,
                alpha=0.7,
            )
        ax.scatter(ray_ends[:, 0], ray_ends[:, 1], color="red", s=20, zorder=5, label="obstacles")

    ax.legend()
    ax.grid(True, alpha=0.3)


def _draw_plane_inliers(ax, depth: np.ndarray, cone_mask: np.ndarray, validation: PlaneValidationResult) -> None:
    rgb_overlay = np.zeros((*depth.shape, 3), dtype=np.uint8)
    cone_pixel_indices = np.argwhere(cone_mask)

    for i, candidate in enumerate(validation.all_candidates):
        color_name = _PLANE_COLORS[i % len(_PLANE_COLORS)]
        color = _color_to_rgb(color_name)
        inlier_pixels = cone_pixel_indices[candidate.plane.inliers]
        rows, cols = inlier_pixels[:, 0], inlier_pixels[:, 1]
        rgb_overlay[rows, cols] = color

    ax.imshow(rgb_overlay)

    selected = validation.best_candidate
    if selected is not None:
        idx = validation.all_candidates.index(selected)
        color = _PLANE_COLORS[idx % len(_PLANE_COLORS)]
        status = "valid" if validation.disqualification_reason is None else f"disqualified: {validation.disqualification_reason}"
        title = f"Planes — selected: plane {idx} ({color})\n{status}"
    else:
        title = f"Planes — none selected\n{validation.disqualification_reason or ''}"

    ax.set_title(title, fontsize=8)
    ax.axis("off")

    handles = []
    for i, candidate in enumerate(validation.all_candidates):
        color = _PLANE_COLORS[i % len(_PLANE_COLORS)]
        patch = plt.Rectangle((0, 0), 1, 1, color=color)
        label = f"P{i}: fill={candidate.cone_fill_fraction:.2f} ang={candidate.normal_angle_deg:.0f}°"
        handles.append((patch, label))

    if handles:
        ax.legend(
            [h for h, _ in handles],
            [l for _, l in handles],
            loc="lower left",
            fontsize=7,
            framealpha=0.7,
        )


def _draw_landmark_plane_debug(ax, agent_x: float, agent_y: float, heading: float, debug_result) -> None:
    ax.set_aspect("equal")
    ax.set_title(
        f"Landmark planes (world 2D)\nfloor alignment: {debug_result.floor_alignment:.2f}"
        if debug_result.floor_alignment is not None else "Landmark planes — no floor detected",
        fontsize=8,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.plot(agent_x, agent_y, "ko", markersize=8, zorder=10)
    dx = _HEADING_ARROW_LENGTH_M * math.cos(heading)
    dy = _HEADING_ARROW_LENGTH_M * math.sin(heading)
    ax.annotate("", xy=(agent_x + dx, agent_y + dy), xytext=(agent_x, agent_y),
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5})

    scatter_colors = ["red", "lime", "cyan"]
    for i, info in enumerate(debug_result.planes):
        color = scatter_colors[i % len(scatter_colors)]
        pts = info.world_2d_points
        sample = pts[::max(1, len(pts) // 500)]  # downsample for speed
        ax.scatter(sample[:, 0], sample[:, 1], s=2, color=color, alpha=0.4, rasterized=True)

        if info.landmark is not None:
            a, b = info.landmark.endpoint_a, info.landmark.endpoint_b
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=3,
                    label=f"P{i} accepted")
        else:
            centroid = pts.mean(axis=0)
            ax.annotate(
                f"P{i}: {info.rejection_reason}",
                xy=centroid,
                fontsize=6,
                color=color,
                ha="center",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7},
            )

    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)


def _draw_all_planes(ax, raw_depth: np.ndarray, intrinsics: Intrinsics) -> None:
    valid_mask = raw_depth > 0
    valid_pixels = np.argwhere(valid_mask)
    rows, cols = valid_pixels[:, 0], valid_pixels[:, 1]
    z = raw_depth[valid_mask].astype(np.float64)
    points = np.stack([
        (cols - intrinsics.cx) * z / intrinsics.fx,
        (rows - intrinsics.cy) * z / intrinsics.fy,
        z,
    ], axis=-1)

    planes = detect_planes(points, n_planes=len(_PLANE_COLORS))

    rgb_overlay = np.zeros((*raw_depth.shape, 3), dtype=np.uint8)
    handles = []
    for i, plane in enumerate(planes):
        color_name = _PLANE_COLORS[i % len(_PLANE_COLORS)]
        rgb_overlay[valid_pixels[plane.inliers, 0], valid_pixels[plane.inliers, 1]] = _color_to_rgb(color_name)
        inlier_count = plane.inliers.sum()
        patch = plt.Rectangle((0, 0), 1, 1, color=color_name)
        handles.append((patch, f"P{i}: {inlier_count} pts ({100 * inlier_count / len(points):.1f}%)"))

    ax.imshow(rgb_overlay)
    ax.set_title("All planes (full image)")
    ax.axis("off")
    if handles:
        ax.legend(
            [h for h, _ in handles],
            [l for _, l in handles],
            loc="lower left",
            fontsize=7,
            framealpha=0.7,
        )


def _color_to_rgb(name: str) -> tuple[int, int, int]:
    mapping = {
        "red": (220, 50, 50),
        "lime": (50, 220, 50),
        "cyan": (50, 220, 220),
    }
    return mapping[name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a saved depth capture")
    parser.add_argument("--path", type=Path, default=_DEFAULT_PATH)
    parser.add_argument("--save", type=Path, default=None, metavar="OUTPUT_PNG")
    args = parser.parse_args()
    _show(args.path, save_path=args.save)
