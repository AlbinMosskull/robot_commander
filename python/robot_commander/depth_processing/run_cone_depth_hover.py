"""
Interactive depth viewer for the cone depth processor.

Shows the calibrated depth image alongside the original frame.
Hover the mouse over the depth image to read out the depth value at that pixel.
The cone region is overlaid in green.

Adjust the cone parameters at the top of this file to match your sensor setup.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from robot_commander.agent.adeept.adeept_transforms import CAMERA_T_SENSOR_CENTER
from robot_commander.depth_processing.cone_depth_processor import (
    ConeDepthProcessor,
    ConeGeometry,
)
from robot_commander.image_processing import intrinsics as intrinsics_io

_IMAGE_PATH = Path("python/robot_commander/depth_processing/tests/test_data/robot_pov_4.jpg")
_INTRINSICS_PATH = Path("calibration/intrinsics.npz")

_ULTRASONIC_READING_M = 0.21

_CONE_HALF_ANGLE_DEGREES = 15.0
def _build_processor() -> ConeDepthProcessor:
    intrinsics = intrinsics_io.load(_INTRINSICS_PATH)
    cone = ConeGeometry(half_angle_radians=np.radians(_CONE_HALF_ANGLE_DEGREES))
    return ConeDepthProcessor(intrinsics=intrinsics, camera_T_sensor=CAMERA_T_SENSOR_CENTER, cone_geometry=cone)


def _depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    colormap_bgr = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)


def main():
    print("Loading model...")
    processor = _build_processor()

    frame_bgr = cv2.imread(str(_IMAGE_PATH))
    if frame_bgr is None:
        raise FileNotFoundError(f"Image not found: {_IMAGE_PATH}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    print("Running depth estimation...")
    depth, cone_mask = processor.process_with_mask(frame_bgr, _ULTRASONIC_READING_M)

    depth_colormap = _depth_to_colormap(depth)

    cone_overlay = depth_colormap.copy()
    cone_overlay[cone_mask] = (
        cone_overlay[cone_mask] * 0.5 + np.array([0, 200, 0]) * 0.5
    ).astype(np.uint8)

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    figure.suptitle(
        f"Cone depth — ultrasonic {_ULTRASONIC_READING_M:.2f} m  |  "
        f"cone half-angle {_CONE_HALF_ANGLE_DEGREES:.0f}°",
        fontsize=13,
    )

    axes[0].imshow(frame_rgb)
    axes[0].set_title("Original frame")
    axes[0].axis("off")

    axes[1].imshow(cone_overlay)
    axes[1].set_title("Calibrated depth  (hover for values)")
    axes[1].axis("off")

    cone_patch = mpatches.Patch(color=(0, 200 / 255, 0), label="Cone region")
    axes[1].legend(handles=[cone_patch], loc="upper right")

    depth_text = axes[1].text(
        0.01, 0.01, "", transform=axes[1].transAxes,
        color="white", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )

    def on_mouse_move(event):
        if event.inaxes is not axes[1]:
            depth_text.set_text("")
            figure.canvas.draw_idle()
            return
        col = int(round(event.xdata)) if event.xdata is not None else None
        row = int(round(event.ydata)) if event.ydata is not None else None
        if col is None or row is None:
            return
        height, width = depth.shape
        if 0 <= row < height and 0 <= col < width:
            value = depth[row, col]
            in_cone = cone_mask[row, col]
            cone_label = "in cone" if in_cone else "outside cone"
            depth_text.set_text(f"({col}, {row})  {value:.3f} m  [{cone_label}]")
            figure.canvas.draw_idle()

    figure.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
