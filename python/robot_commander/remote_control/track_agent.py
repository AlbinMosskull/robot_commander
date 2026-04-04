import cv2

from robot_commander.remote_control.agent_client import AgentClient
from robot_commander.remote_control.stencil_map_controller import StencilMapController


def main():
    client = AgentClient()
    controller = StencilMapController(client)
    controller.start()

    def on_mouse(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        shift_held = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        controller.handle_click(x, y, shift_held)

    cv2.namedWindow("Agent Map")
    cv2.setMouseCallback("Agent Map", on_mouse)

    print("Left-click: set checkpoint. Shift+click: plan path. Press 'q' to quit.")
    try:
        while True:
            canvas = controller.render()
            cv2.imshow("Agent Map", canvas)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
