import math

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from robot_commander.depth_processing.depth_capture import DepthCapture
from robot_commander.remote_control.controller import RemoteControl

_TOP_DOWN_SIZE = 300
_TOP_DOWN_RANGE_M = 2.0
_HEADING_ARROW_PX = 30


def _numpy_bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)


def _render_depth_map(depth: np.ndarray, cone_mask: np.ndarray) -> np.ndarray:
    masked = np.where(cone_mask, depth, 0.0).astype(np.float32)
    valid = masked[cone_mask]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    normalized = np.clip(masked / valid.max(), 0.0, 1.0)
    gray = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_PLASMA)


def _render_frame_with_cone_mask(frame: np.ndarray, cone_mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    green_tint = np.array([0, 180, 0], dtype=np.uint8)
    overlay[cone_mask] = (overlay[cone_mask] * 0.5 + green_tint * 0.5).astype(np.uint8)
    return overlay


def _render_top_down_rays(capture: DepthCapture) -> np.ndarray:
    canvas = np.zeros((_TOP_DOWN_SIZE, _TOP_DOWN_SIZE, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    scale = _TOP_DOWN_SIZE / (2.0 * _TOP_DOWN_RANGE_M)
    cx = _TOP_DOWN_SIZE // 2
    cy = _TOP_DOWN_SIZE // 2

    def world_to_px(wx: float, wy: float) -> tuple[int, int]:
        px = int(cx + (wx - capture.agent_x) * scale)
        py = int(cy - (wy - capture.agent_y) * scale)
        return px, py

    agent_px = world_to_px(capture.agent_x, capture.agent_y)

    for end_x, end_y in capture.ray_ends:
        end_px = world_to_px(end_x, end_y)
        cv2.line(canvas, agent_px, end_px, (60, 120, 200), 1, cv2.LINE_AA)

    if len(capture.ray_ends) > 0:
        for end_x, end_y in capture.ray_ends:
            end_px = world_to_px(end_x, end_y)
            cv2.circle(canvas, end_px, 3, (50, 50, 220), -1, cv2.LINE_AA)

    arrow_dx = int(_HEADING_ARROW_PX * math.cos(capture.heading))
    arrow_dy = int(-_HEADING_ARROW_PX * math.sin(capture.heading))
    cv2.arrowedLine(canvas, agent_px, (agent_px[0] + arrow_dx, agent_px[1] + arrow_dy),
                    (200, 180, 50), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(canvas, agent_px, 5, (200, 180, 50), -1, cv2.LINE_AA)

    return canvas


def _composite(capture: DepthCapture, target_width: int, target_height: int) -> np.ndarray:
    frame_view = _render_frame_with_cone_mask(capture.frame, capture.cone_mask)
    depth_view = _render_depth_map(capture.calibrated_depth, capture.cone_mask)
    rays_view = _render_top_down_rays(capture)

    panel_w = target_width // 3
    panel_h = target_height

    def resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (panel_w, panel_h))

    return np.concatenate([resize(frame_view), resize(depth_view), resize(rays_view)], axis=1)


class DepthWidget(QWidget):
    def __init__(self, controller: RemoteControl, parent=None):
        super().__init__(parent)
        self._controller = controller
        self.setStyleSheet("background-color: #0d0d0d;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        panel_label = QLabel("DEPTH VIEW")
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
        capture = self._controller.latest_depth_capture
        if capture is None:
            return
        w = self._display.width()
        h = self._display.height()
        if w <= 0 or h <= 0:
            return
        composite = _composite(capture, w, h)
        self._display.setPixmap(
            _numpy_bgr_to_pixmap(composite).scaled(
                self._display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
