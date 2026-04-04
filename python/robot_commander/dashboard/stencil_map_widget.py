import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from robot_commander.image_processing.camera import Camera
from robot_commander.remote_control.stencil_map_controller import StencilMapController


def _numpy_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)


class StencilMapWidget(QWidget):
    def __init__(self, controller: StencilMapController, camera: Camera, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._camera = camera
        self.setStyleSheet("background-color: #0d0d0d;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        panel_label = QLabel("MINIMAP")
        panel_label.setFixedHeight(20)
        panel_label.setStyleSheet(
            "color: #888888; font-family: monospace; font-size: 11px;"
            "background-color: #111111; padding-left: 6px;"
        )

        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setStyleSheet("background-color: #0d0d0d;")
        self._display.setMouseTracking(True)

        layout.addWidget(panel_label)
        layout.addWidget(self._display, stretch=1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)

    def _refresh(self) -> None:
        ok, frame = self._camera.read()
        if ok:
            self._controller.update(frame)

        canvas = self._controller.render()
        pixmap = _numpy_to_pixmap(canvas)
        self._display.setPixmap(
            pixmap.scaled(
                self._display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        # Map click position from widget coords to original frame pixel coords.
        frame_w, frame_h = self._controller.frame_size
        display_size = self._display.size()

        scale = min(display_size.width() / frame_w, display_size.height() / frame_h)
        scaled_w = int(frame_w * scale)
        scaled_h = int(frame_h * scale)

        # Offset of the scaled image within the QLabel (centered).
        offset_x = (display_size.width() - scaled_w) // 2
        offset_y = (display_size.height() - scaled_h) // 2

        # Position relative to this widget, adjusted for the panel label height.
        label_height = 20
        click_x = event.position().x()
        click_y = event.position().y() - label_height

        # Convert to frame pixel coords.
        frame_px_x = int((click_x - offset_x) / scale)
        frame_px_y = int((click_y - offset_y) / scale)

        if 0 <= frame_px_x < frame_w and 0 <= frame_px_y < frame_h:
            shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self._controller.handle_click(frame_px_x, frame_px_y, shift_held)
