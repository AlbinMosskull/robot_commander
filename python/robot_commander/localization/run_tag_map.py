"""
Live top-down map of detected AprilTag positions overlaid on the stencil map.

Loads output/debug/09_stencil_map.png as the background.  Each detected tag
is drawn as a labelled dot using the same coordinate projection as the stencil
map (_to_map_px from main.py).  A faded trail shows the last N positions.

Press 'q' to quit.
"""

import collections
from pathlib import Path

import cv2
import numpy as np

from robot_commander.camera import intrinsics as cal
from robot_commander.camera.camera import Camera
from robot_commander.camera.tag_detector import TagDetector, draw_tags
from robot_commander.config import load as load_config
from robot_commander.localization.localizer import Localizer
from robot_commander.remote_control.main import _to_floor_2d, _to_map_px

_cfg = load_config()
_TRAIL_LEN = 30    # past positions to show per tag ID

_STENCIL_PATH = Path("output/debug/09_stencil_map.png")
_BASIS_PATH   = Path("output/debug/floor_basis.npz")

_TAG_COLORS = [
    (0,  200, 255),
    (0,  255, 100),
    (255, 100,  0),
    (200,   0, 255),
    (255, 220,   0),
    (0,  160, 255),
]


def _to_px(
    pos: tuple[float, float, float],
    u_vec: np.ndarray,
    v_vec: np.ndarray,
) -> tuple[int, int]:
    """Project a camera-space 3-D position onto the stencil map."""
    floor_2d = _to_floor_2d(np.array([pos]), u_vec, v_vec)
    px = _to_map_px(floor_2d)[0]
    return int(px[0]), int(px[1])


def _draw_tags_on_map(
    base: np.ndarray,
    trails: dict[int, collections.deque],
    current: list[tuple[int, tuple[float, float, float]]],
    u_vec: np.ndarray,
    v_vec: np.ndarray,
) -> np.ndarray:
    canvas = base.copy()

    for tag_id, pos_deque in trails.items():
        color = _TAG_COLORS[tag_id % len(_TAG_COLORS)]
        pts = list(pos_deque)
        for i, pos in enumerate(pts):
            alpha = (i + 1) / len(pts)
            faded = tuple(int(c * alpha * 0.4) for c in color)
            cv2.circle(canvas, _to_px(pos, u_vec, v_vec), 3, faded, -1)

    for tag_id, pos in current:
        color = _TAG_COLORS[tag_id % len(_TAG_COLORS)]
        mp = _to_px(pos, u_vec, v_vec)
        dist = float(np.linalg.norm(pos))
        cv2.circle(canvas, mp, 9, color, -1)
        cv2.circle(canvas, mp, 9, (0, 0, 0), 1)
        label = f"ID {tag_id}  {dist:.2f}m"
        cv2.putText(canvas, label, (mp[0] + 12, mp[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if not current:
        h = base.shape[0]
        cv2.putText(canvas, "No tags detected", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

    return canvas


def main():
    for path, label in [(_STENCIL_PATH, "stencil map"), (_BASIS_PATH, "floor basis")]:
        if not path.exists():
            print(f"{label.capitalize()} not found at {path}.")
            print("Run debug_stencil.py first to generate it.")
            return

    stencil_base = cv2.imread(str(_STENCIL_PATH))
    basis = np.load(_BASIS_PATH)
    u_vec, v_vec = basis["u_vec"], basis["v_vec"]

    intrinsics = cal.load()
    detector = TagDetector()
    localizer = Localizer(detector, intrinsics.camera_matrix, _cfg.tag.size_m,
                          dist_coeffs=intrinsics.dist_coeffs)

    trails: dict[int, collections.deque] = {}

    with Camera() as cam:
        print("Camera opened. Press 'q' to quit.")
        cam.warm_up()

        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            detections = localizer.localize_all(frame)

            # All detections individually — multiple tags with same ID each plotted.
            current = [(tag.tag_id, pos) for tag, pos in detections]
            for tag_id, pos in current:
                if tag_id not in trails:
                    trails[tag_id] = collections.deque(maxlen=_TRAIL_LEN)
                trails[tag_id].append(pos)

            annotated = draw_tags(frame, [t for t, _ in detections])
            small = cv2.resize(annotated, (0, 0), fx=0.5, fy=0.5)
            map_img = _draw_tags_on_map(stencil_base, trails, current, u_vec, v_vec)

            cv2.imshow("Camera", small)
            cv2.imshow("Tag Map", map_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
