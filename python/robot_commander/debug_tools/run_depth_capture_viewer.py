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

_DEFAULT_PATH = Path(__file__).parent / "latest_depth_capture.npz"
_HEADING_ARROW_LENGTH_M = 0.3
_cfg = load_config()


def _build_depth_processor(capture) -> ConeDepthProcessor:
    cone_geometry = ConeGeometry(half_angle_radians=math.radians(_cfg.depth.cone_half_angle_deg))
    return ConeDepthProcessor(
        intrinsics=capture.intrinsics,
        camera_T_sensor=CAMERA_T_SENSOR_CENTER,
        cone_geometry=cone_geometry,
    )


def _show(
    capture_path: Path,
    save_path: Path | None = None,
) -> None:
    capture = load(capture_path)

    depth_processor = _build_depth_processor(capture)
    calibrated_depth, cone_mask = depth_processor.process_with_mask(capture.frame, capture.ultrasonic_min)

    print(f"agent: ({capture.agent_x:.4f}, {capture.agent_y:.4f})  heading: {capture.heading:.4f} rad")
    print(f"calibrated_depth range: [{calibrated_depth.min():.4f}, {calibrated_depth.max():.4f}] m")
    print(f"calibrated_depth (cone only) range: [{calibrated_depth[cone_mask].min():.4f}, {calibrated_depth[cone_mask].max():.4f}] m")

    masked_depth = np.where(cone_mask, calibrated_depth, 0.0).astype(np.float32)
    rays = depth_to_rays(masked_depth, capture.intrinsics, capture.agent_x, capture.agent_y, capture.heading)
    ray_ends = rays_to_ends(rays)
    print(f"ray_ends (regenerated): {ray_ends.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Depth capture — {capture_path.name}")

    _draw_frame_with_cone_mask(axes[0], capture.frame, cone_mask)
    _draw_depth_map(axes[1], calibrated_depth)
    _draw_top_down_rays(axes[2], capture.agent_x, capture.agent_y, capture.heading, ray_ends)

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


def _draw_depth_map(ax, depth: np.ndarray) -> None:
    im = ax.imshow(depth, cmap="plasma", vmin=0)
    plt.colorbar(im, ax=ax, label="depth (m)")
    ax.set_title("Calibrated depth")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a saved depth capture")
    parser.add_argument("--path", type=Path, default=_DEFAULT_PATH)
    parser.add_argument("--save", type=Path, default=None, metavar="OUTPUT_PNG")
    args = parser.parse_args()
    _show(args.path, save_path=args.save)
