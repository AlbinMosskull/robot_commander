from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from robot_commander.dashboard.qt_image_utils import numpy_bgr_to_pixmap
from robot_commander.image_processing.camera import Camera


class CameraWidget(QWidget):
    def __init__(self, camera: Camera, label: str, parent=None):
        super().__init__(parent)
        self._camera = camera
        self.setStyleSheet("background-color: #0d0d0d;")

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        panel_label = QLabel(label)
        panel_label.setFixedHeight(20)
        panel_label.setStyleSheet(
            "color: #888888; font-family: monospace; font-size: 11px;"
            "background-color: #111111; padding-left: 6px;"
        )

        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setStyleSheet("background-color: #0d0d0d;")

        outer_layout.addWidget(panel_label)
        outer_layout.addWidget(self._display, stretch=1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)

    def _refresh(self) -> None:
        ok, frame = self._camera.read()
        if not ok:
            return
        pixmap = numpy_bgr_to_pixmap(frame)
        self._display.setPixmap(
            pixmap.scaled(
                self._display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def stop(self) -> None:
        self._timer.stop()
        self._camera.release()
