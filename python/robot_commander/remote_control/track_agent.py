import threading
from pathlib import Path

import cv2

from robot_commander.map_building.map_coordinates import px_to_world, world_to_px
from robot_commander.remote_control.agent_client import AgentClient

_STENCIL_PATH = Path("plots/output/stencil_map.png")


def main():
    if not _STENCIL_PATH.exists():
        print(f"Stencil map not found at {_STENCIL_PATH}. Run build_stencil_map.py first.")
        return

    background = cv2.imread(str(_STENCIL_PATH))
    client = AgentClient()

    checkpoint: tuple[float, float] | None = None
    agent_pos: tuple[float, float] | None = None
    pos_lock = threading.Lock()

    def on_mouse(event, x, y, flags, param):
        nonlocal checkpoint
        if event == cv2.EVENT_LBUTTONDOWN:
            wx, wy = px_to_world(x, y)
            checkpoint = (wx, wy)
            client.set_checkpoint(wx, wy)

    def stream_thread():
        nonlocal agent_pos
        for x, y in client.stream_positions():
            with pos_lock:
                agent_pos = (x, y)

    t = threading.Thread(target=stream_thread, daemon=True)
    t.start()

    cv2.namedWindow("Agent Map")
    cv2.setMouseCallback("Agent Map", on_mouse)

    print("Click on the map to place a checkpoint. Press 'q' to quit.")
    while True:
        canvas = background.copy()

        if checkpoint is not None:
            px = world_to_px(*checkpoint)
            cv2.circle(canvas, px, 8, (0, 200, 0), -1)
            cv2.circle(canvas, px, 8, (0, 0, 0), 1)

        with pos_lock:
            pos = agent_pos
        if pos is not None:
            px = world_to_px(*pos)
            cv2.circle(canvas, px, 8, (200, 80, 0), -1)
            cv2.circle(canvas, px, 8, (0, 0, 0), 1)

        cv2.imshow("Agent Map", canvas)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    client.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
