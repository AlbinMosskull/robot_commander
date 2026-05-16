import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from robot_commander.config import load as load_config
from robot_commander.agent.adeept.adeept_transforms import CAMERA_T_SENSOR_CENTER
from robot_commander.depth_processing.cone_depth_processor import ConeDepthProcessor, ConeGeometry
from robot_commander.depth_processing.depth_rays import depth_to_rays, floor_plane_basis
from robot_commander.depth_processing.depth_capture import load, rays_to_ends
from robot_commander.depth_processing.point_cloud import depth_image_to_point_cloud
from robot_commander.depth_processing.ransac import detect_floor, detect_planes
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


def _print_ray_pipeline_diagnostics(calibrated_depth: np.ndarray, intrinsics) -> None:
    from robot_commander.depth_processing.depth_rays import _MIN_OBSTACLE_HEIGHT_M
    print("\n--- Ray pipeline diagnostics ---")
    camera_points = depth_image_to_point_cloud(calibrated_depth, intrinsics)
    print(f"  point cloud size: {len(camera_points)}")
    ones = np.ones((len(camera_points), 1), dtype=np.float32)
    points = np.hstack([camera_points, ones])[:, :3]
    floor = detect_floor(points)
    if floor is None:
        print("  floor: NOT DETECTED — no rays possible")
        return
    print(f"  floor normal: {floor.normal}  distance: {floor.distance:.4f}  inliers: {floor.inliers.sum()}")
    heights = points @ floor.normal - floor.distance
    above = (heights > _MIN_OBSTACLE_HEIGHT_M)
    print(f"  points above floor (>{_MIN_OBSTACLE_HEIGHT_M} m): {above.sum()} / {len(points)}")
    right, forward = floor_plane_basis(floor.normal)
    valid = points[above]
    if len(valid) > 0:
        rights = valid @ right
        forwards = valid @ forward
        horiz = np.sqrt(rights ** 2 + forwards ** 2)
        print(f"  horizontal distances: min={horiz.min():.3f}  p5={np.percentile(horiz, 5):.3f}  max={horiz.max():.3f} m")
    print()


def _show(
    capture_path: Path,
    save_path: Path | None = None,
    no_plot: bool = False,
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
    if len(rays) > 0:
        lengths = [math.hypot(r.end_x - r.start_x, r.end_y - r.start_y) for r in rays]
        print(f"ray lengths: min={min(lengths):.3f}  median={sorted(lengths)[len(lengths)//2]:.3f}  max={max(lengths):.3f} m")
    _print_ray_pipeline_diagnostics(calibrated_depth, capture.intrinsics)

    if no_plot:
        return

    display_depth = calibrated_depth if capture.is_calibrated else raw_depth
    calibration_tag = "calibrated" if capture.is_calibrated else "uncalibrated (raw metric)"

    fig, axes = plt.subplots(1, 6, figsize=(33, 5))
    fig.suptitle(f"Depth capture — {capture_path.name}  [{calibration_tag}]")

    _draw_frame_with_cone_mask(axes[0], capture.frame, cone_mask)
    _draw_depth_map(axes[1], display_depth, title=f"Depth ({calibration_tag})")
    _draw_top_down_rays(axes[2], capture.agent_x, capture.agent_y, capture.heading, ray_ends)
    _draw_plane_inliers(axes[3], display_depth, validation_mask, validation)
    _draw_all_planes(axes[4], raw_depth, capture.intrinsics)

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
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    _show(args.path, save_path=args.save, no_plot=args.no_plot)
