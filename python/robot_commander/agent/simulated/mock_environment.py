from pathlib import Path

import cv2
import numpy as np

from robot_commander.config import load as load_config
from robot_commander.map_building.map_coordinates import MapCoordinates

_OBSTACLES_PATH = Path(__file__).parent / "obstacles.png"
_OBSTACLE_RADIUS_PX = 10


def main():
    stencil_path = load_config().map.stencil_path
    if not stencil_path.exists():
        print(f"Stencil map not found at {stencil_path}. Run build_stencil_map.py first.")
        return

    map_coords = MapCoordinates.load(stencil_path)
    background = map_coords.background
    obstacles = np.zeros((map_coords.height_px, map_coords.width_px), dtype=np.uint8)

    if _OBSTACLES_PATH.exists():
        loaded = cv2.imread(str(_OBSTACLES_PATH), cv2.IMREAD_GRAYSCALE)
        if loaded is not None and loaded.shape == obstacles.shape:
            obstacles = loaded
            print(f"Loaded existing obstacles from {_OBSTACLES_PATH}")

    painting = False

    def on_mouse(event, x, y, flags, param):
        nonlocal painting
        if event == cv2.EVENT_LBUTTONDOWN:
            painting = True
        elif event == cv2.EVENT_LBUTTONUP:
            painting = False
        if painting and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            cv2.circle(obstacles, (x, y), _OBSTACLE_RADIUS_PX, 255, -1)

    cv2.namedWindow("Mock Environment")
    cv2.setMouseCallback("Mock Environment", on_mouse)

    print("Left-click/drag to add obstacles.")
    print("  's' — save  |  'c' — clear  |  'q' — quit")

    while True:
        canvas = background.copy()
        canvas[obstacles > 0] = (0, 0, 200)
        cv2.imshow("Mock Environment", canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(str(_OBSTACLES_PATH), obstacles)
            print(f"Saved obstacles to {_OBSTACLES_PATH}")
        elif key == ord('c'):
            obstacles[:] = 0
            print("Cleared all obstacles")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
