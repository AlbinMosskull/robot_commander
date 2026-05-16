import math
import threading
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from robot_commander.dashboard.agent_camera import AgentCamera, OverheadCamera
from robot_commander.dashboard.camera_widget import CameraWidget
from robot_commander.dashboard.depth_widget import DepthWidget
from robot_commander.dashboard.map_widget import MapWidget
from robot_commander.dashboard.payload_widget import PayloadWidget
from robot_commander.dashboard.status_bar import StatusBarWidget
from robot_commander.image_processing.camera import FromFileCamera, WebCamera
from robot_commander.remote_control.agent_client import AgentClient
from robot_commander.remote_control.controller import build_controller

_EXAMPLE_INPUT = Path("images/example_input")


def _try_connect_client(host: str | None = None) -> AgentClient | None:
    try:
        return AgentClient(host=host) if host is not None else AgentClient()
    except Exception:
        return None


class DashboardWindow(QMainWindow):
    def __init__(self, show_escape_plan: bool = False, agent_host: str | None = None):
        super().__init__()
        self.setWindowTitle("Robot Commander")
        self.resize(1280, 720)
        self.setStyleSheet(
            "background-color: #1a1a1a; color: #e0e0e0; font-family: monospace;"
        )

        self._client = _try_connect_client(agent_host)
        simulated = agent_host is not None
        camera_overhead = FromFileCamera(_EXAMPLE_INPUT / "scene_image.jpg") if simulated else WebCamera()
        self._controller = build_controller(self._client, camera_overhead)

        camera_pov = AgentCamera(self._controller) if self._client is not None else FromFileCamera(_EXAMPLE_INPUT / "robot_pov.jpg")

        root_widget = QWidget()
        self.setCentralWidget(root_widget)

        root_layout = QVBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(StatusBarWidget(self._controller))

        main_area = QWidget()
        main_layout = QHBoxLayout(main_area)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        self._map_widget = MapWidget(self._controller, show_escape_plan=show_escape_plan)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self._cam_overhead_widget = CameraWidget(OverheadCamera(self._controller), "OVERHEAD ANGLE (CAM-02)")
        left_layout.addWidget(self._map_widget, stretch=1)
        left_layout.addWidget(self._cam_overhead_widget, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self._cam_pov_widget = CameraWidget(camera_pov, "POV CAMERA (CAM-01)")
        self._depth_widget = DepthWidget(self._controller)
        self._payload_widget = PayloadWidget(self._controller)

        self._payload_widget.setMaximumHeight(120)
        right_layout.addWidget(self._cam_pov_widget, stretch=1)
        right_layout.addWidget(self._depth_widget, stretch=1)
        right_layout.addWidget(self._payload_widget)

        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=1)

        root_layout.addWidget(main_area, stretch=1)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Q:
            self._controller.set_offset_waypoint(math.pi / 2, 0.20)
        elif event.key() == Qt.Key.Key_S and self._client is not None:
            threading.Thread(target=self._client.scout, daemon=True).start()
        elif event.key() == Qt.Key.Key_P and self._client is not None:
            threading.Thread(target=self._controller.enable_payload, daemon=True).start()
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._controller.start()

    def closeEvent(self, event) -> None:
        self._controller.stop()
        self._cam_pov_widget.stop()
        self._cam_overhead_widget.stop()
        super().closeEvent(event)
