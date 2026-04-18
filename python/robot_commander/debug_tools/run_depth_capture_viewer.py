import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from robot_commander.depth_processing.depth_capture import load

_DEFAULT_PATH = Path(__file__).parent / "latest_depth_capture.npz"
_HEADING_ARROW_LENGTH_M = 0.3


def _show(capture_path: Path) -> None:
    capture = load(capture_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Depth capture — {capture_path.name}")

    _draw_frame_with_cone_mask(axes[0], capture.frame, capture.cone_mask)
    _draw_depth_map(axes[1], capture.calibrated_depth)
    _draw_top_down_rays(axes[2], capture)

    plt.tight_layout()
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


def _draw_top_down_rays(ax, capture) -> None:
    ax.set_aspect("equal")
    ax.set_title("Top-down rays (world frame, m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.plot(capture.agent_x, capture.agent_y, "bo", markersize=8, label="agent")

    dx = _HEADING_ARROW_LENGTH_M * math.cos(capture.heading)
    dy = _HEADING_ARROW_LENGTH_M * math.sin(capture.heading)
    ax.annotate(
        "",
        xy=(capture.agent_x + dx, capture.agent_y + dy),
        xytext=(capture.agent_x, capture.agent_y),
        arrowprops={"arrowstyle": "->", "color": "blue", "lw": 1.5},
    )

    if len(capture.ray_ends) > 0:
        for end_x, end_y in capture.ray_ends:
            ax.plot(
                [capture.agent_x, end_x],
                [capture.agent_y, end_y],
                color="orange",
                linewidth=1,
                alpha=0.7,
            )
        ax.scatter(capture.ray_ends[:, 0], capture.ray_ends[:, 1], color="red", s=20, zorder=5, label="obstacles")

    ax.legend()
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a saved depth capture")
    parser.add_argument("--path", type=Path, default=_DEFAULT_PATH)
    args = parser.parse_args()
    _show(args.path)
