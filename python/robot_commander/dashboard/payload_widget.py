import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from robot_commander.remote_control.controller import RemoteControl


def _no_payload_frame(w: int, h: int) -> np.ndarray:
    canvas = np.full((h, w, 3), 13, dtype=np.uint8)
    text = "NO PAYLOAD"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(canvas, text, ((w - tw) // 2, (h + th) // 2), font, scale, (80, 80, 80), thickness)
    return canvas


def _numpy_bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)


class PayloadWidget(QWidget):
    def __init__(self, controller: RemoteControl, parent=None):
        super().__init__(parent)
        self._controller = controller
        self.setStyleSheet("background-color: #0d0d0d;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        panel_label = QLabel("PAYLOAD IMAGE")
        panel_label.setFixedHeight(20)
        panel_label.setStyleSheet(
            "color: #888888; font-family: monospace; font-size: 11px;"
            "background-color: #111111; padding-left: 6px;"
        )

        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setStyleSheet("background-color: #0d0d0d;")

        layout.addWidget(panel_label)
        layout.addWidget(self._display, stretch=1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(200)

    def _refresh(self) -> None:
        w = self._display.width()
        h = self._display.height()
        if w <= 0 or h <= 0:
            return
        frame = self._controller.latest_payload_frame
        if frame is None:
            pixmap = _numpy_bgr_to_pixmap(_no_payload_frame(w, h))
        else:
            pixmap = _numpy_bgr_to_pixmap(frame)
        self._display.setPixmap(
            pixmap.scaled(
                self._display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
