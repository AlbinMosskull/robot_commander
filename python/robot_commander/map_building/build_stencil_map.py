"""
Builds the stencil map.
"""

import time
from pathlib import Path
import argparse

import cv2
import numpy as np

from robot_commander.image_processing import intrinsics as cal
from robot_commander.image_processing.camera import Camera, FromFileCamera, WebCamera
from robot_commander.config import load as load_config
from robot_commander.image_processing.tag_detector import TagDetector
from robot_commander.depth_processing.calibrated_depth_processor import CalibratedDepthProcessor
from robot_commander.localization.localizer import Localizer
from robot_commander.semantic_understanding.detection_segmentor import DetectionSegmentor
from robot_commander.config import load as load_config
from robot_commander.map_building.map_coordinates import MapCoordinates
from robot_commander.map_building.map_drawing import draw_stencil_map
from robot_commander.map_building.map_geometry import FootprintResult, to_floor_2d, build_footprints, detect_floor
from robot_commander.map_building.debug_map_building import (
    check_depth_and_save_vis,
    save_mask_vis,
    save_scatter,
    check_tag_normals,
    save_surface_vis,
    save_floor_vis,
)

_cfg = load_config()
_DEBUG_DIR = Path("plots/debug")

_OBJECT_CLASSES: dict[str, str] = {
    "dining table": "dining table",
    "couch":        "couch",
}
_NUM_FRAMES = 5
_FRAME_INTERVAL_S = 0.3

_CLASS_COLORS_BGR = {
    "dining table": (0,  80, 220),  # red-ish
    "couch":        (0, 180,  60),  # green-ish
}

_TARGET_LABELS = set(_OBJECT_CLASSES.values())



def _gather_frames(cam: Camera, depth_processor: CalibratedDepthProcessor, localizer: Localizer) -> tuple[list[np.ndarray], np.ndarray]:
    with cam:
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

    return frames, depth0


def build_stencil_map(cam: Camera, plot_debug: bool = False) -> MapCoordinates:
    print("Loading models...")
    intrinsics = cal.load()
    detector_model = TagDetector()
    localizer = Localizer(detector_model, intrinsics.camera_matrix, _cfg.tag.size_m,
                          dist_coeffs=intrinsics.dist_coeffs)
    depth_processor = CalibratedDepthProcessor()
    segmentor = DetectionSegmentor()

    print("Gathering frames...")
    frames, depth0 = _gather_frames(cam, depth_processor, localizer)
    print("\nDetecting object masks...")
    frame_masks = {
        r.label: r.mask
        for r in segmentor.process(frames[0])
        if r.label in _TARGET_LABELS
    }
    print(f"  Detected classes: {list(frame_masks.keys())}")

    if plot_debug:
        cv2.imwrite(str(_DEBUG_DIR / "01_frame.jpg"), frames[0])
        check_depth_and_save_vis(frames[0], depth0, _DEBUG_DIR / "02_depth.png")
        save_mask_vis(frames[0], frame_masks, _CLASS_COLORS_BGR, _DEBUG_DIR / "03_roi_masks.jpg")

    floor = detect_floor(depth0, intrinsics)
    if floor is None:
        raise SystemExit("Floor plane detection failed, cannot proceed.")

    if plot_debug:
        save_floor_vis(frames[0], depth0, floor, _DEBUG_DIR / "04_floor_ransac.jpg")
        check_tag_normals(localizer, frames[0], floor.normal)

    map_coords = MapCoordinates.default()
    depths = [depth0] + [depth_processor.process(f) for f in frames[1:]]
    result: FootprintResult = build_footprints(depths, frame_masks, intrinsics, floor.normal, floor.distance, map_coords)

    if plot_debug:
        print(f"\n[PER-CLASS SURFACE]")
        for canonical, pts_3d in result.label_points.items():
            fname = f"05_{canonical.replace(' ', '_')}_surface.jpg"
            save_surface_vis(frames[0], pts_3d, intrinsics, canonical, _DEBUG_DIR / fname)

        save_scatter(result.footprints, _DEBUG_DIR / "06_scatter.png")

        print(f"\n[SHADOW] camera height: {abs(floor.distance):.3f} m  |  surface heights: "
            + ", ".join(f"{k}: {v:.3f} m" for k, v in result.surface_heights.items()))

    map_coords.u_floor = result.u_floor
    map_coords.v_floor = result.v_floor
    map_coords.background = draw_stencil_map(result.footprints, intrinsics, abs(floor.distance), result.surface_heights, map_coords)
    return map_coords


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="webcam",
                        help="Camera source: 'webcam' or path to an image file for testing.")
    parser.add_argument("--plot_debug", action="store_true", help="Whether to save debug plots to disk.")
    args = parser.parse_args()

    cam = WebCamera() if args.camera == "webcam" else FromFileCamera(Path(args.camera))

    stencil_path = load_config().map.stencil_path
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stencil_path.parent.mkdir(parents=True, exist_ok=True)

    map_coords = build_stencil_map(cam, args.plot_debug)
    map_coords.save(stencil_path)



if __name__ == "__main__":
    _main()