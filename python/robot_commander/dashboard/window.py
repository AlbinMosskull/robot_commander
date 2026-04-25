import math
import threading
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from robot_commander.config import load as load_config
from robot_commander.dashboard.agent_camera import AgentCamera
from robot_commander.dashboard.camera_widget import CameraWidget
from robot_commander.dashboard.depth_widget import DepthWidget
from robot_commander.dashboard.map_widget import MapWidget
from robot_commander.dashboard.payload_widget import PayloadWidget
from robot_commander.dashboard.status_bar import StatusBarWidget
from robot_commander.agent.adeept.adeept_transforms import CAMERA_T_SENSOR_CENTER
from robot_commander.depth_processing.cone_depth_processor import ConeDepthProcessor, ConeGeometry
from robot_commander.image_processing import intrinsics as calibration
from robot_commander.image_processing.camera import FromFileCamera, WebCamera
from robot_commander.image_processing.intrinsics import AGENT_CAMERA_PATH, Intrinsics
from robot_commander.image_processing.tag_detector import TagDetector
from robot_commander.localization.camera_localizer import CameraLocalizer
from robot_commander.localization.localizer import Localizer
from robot_commander.map.map_coordinates import MapCoordinates
from robot_commander.remote_control.agent_client import AgentClient
from robot_commander.remote_control.controller import RemoteControl

_EXAMPLE_INPUT = Path("images/example_input")
_cfg = load_config()


def _try_connect_client() -> AgentClient | None:
    try:
        return AgentClient()
    except Exception:
        return None


def _build_localizer_and_depth_processor(
    overhead_intrinsics: Intrinsics,
    agent_intrinsics: Intrinsics,
) -> tuple[CameraLocalizer, ConeDepthProcessor]:
    detector = TagDetector()
    localizer = Localizer(detector, overhead_intrinsics.camera_matrix, _cfg.tag.size_m,
                          dist_coeffs=overhead_intrinsics.dist_coeffs)
    map_coords = MapCoordinates.load(_cfg.map.stencil_path)
    heading_offset = math.radians(_cfg.localization.heading_offset_deg)
    camera_localizer = CameraLocalizer(localizer, map_coords, heading_offset=heading_offset)

    cone_geometry = ConeGeometry(half_angle_radians=math.radians(_cfg.depth.cone_half_angle_deg))
    depth_processor = ConeDepthProcessor(
        intrinsics=agent_intrinsics,
        camera_T_sensor=CAMERA_T_SENSOR_CENTER,
        cone_geometry=cone_geometry,
    )
    return camera_localizer, depth_processor
    # return camera_localizer, None


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Commander")
        self.resize(1280, 720)
        self.setStyleSheet(
            "background-color: #1a1a1a; color: #e0e0e0; font-family: monospace;"
        )

        self._client = _try_connect_client()
        if self._client is not None:
            overhead_intrinsics = calibration.load()
            agent_intrinsics = calibration.load(AGENT_CAMERA_PATH)
            localizer, depth_processor = _build_localizer_and_depth_processor(overhead_intrinsics, agent_intrinsics)
            self._controller = RemoteControl(self._client, localizer,
                                             cone_depth_processor=depth_processor,
                                             cone_intrinsics=agent_intrinsics)
        else:
            self._controller = RemoteControl(None, None)

        camera_pov = AgentCamera(self._controller) if self._client is not None else FromFileCamera(_EXAMPLE_INPUT / "robot_pov.jpg")
        camera_overhead = WebCamera()

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

        self._map_widget = MapWidget(self._controller, camera_overhead)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self._cam_overhead_widget = CameraWidget(camera_overhead, "OVERHEAD ANGLE (CAM-02)")
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
