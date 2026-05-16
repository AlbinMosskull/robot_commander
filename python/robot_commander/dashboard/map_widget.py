import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

import cv2

from robot_commander.dashboard.map_renderer import MapRenderer
from robot_commander.dashboard.qt_image_utils import numpy_bgr_to_pixmap
from robot_commander.remote_control.controller import RemoteControl

_DRAG_HEADING_THRESHOLD_PX = 8


class MapWidget(QWidget):
    def __init__(self, controller: RemoteControl, show_escape_plan: bool = False, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._renderer = MapRenderer(controller.map_coords, show_escape_plan=show_escape_plan)
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

        self._drag_start_frame_px: tuple[int, int] | None = None
        self._drag_current_frame_px: tuple[int, int] | None = None

    def _refresh(self) -> None:
        canvas = self._renderer.render(self._controller.snapshot())

        if self._drag_start_frame_px is not None and self._drag_current_frame_px is not None:
            self._draw_drag_preview(canvas)

        self._display.setPixmap(
            numpy_bgr_to_pixmap(canvas).scaled(
                self._display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _draw_drag_preview(self, canvas: np.ndarray) -> None:
        start = self._drag_start_frame_px
        current = self._drag_current_frame_px
        cv2.arrowedLine(canvas, start, current, (0, 200, 0), 2, cv2.LINE_AA, tipLength=0.2)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = self._frame_px_from_event(event)
        if pos is None:
            return
        shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if shift_held:
            self._drag_start_frame_px = pos
            self._drag_current_frame_px = pos
        else:
            self._drag_start_frame_px = None
            self._drag_current_frame_px = None
            self._controller.handle_click(*pos, shift_held=False)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start_frame_px is None:
            return
        pos = self._frame_px_from_event(event)
        if pos is not None:
            self._drag_current_frame_px = pos

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start_frame_px is None:
            return
        start = self._drag_start_frame_px
        end = self._drag_current_frame_px or start
        goal_heading = self._heading_from_drag(start, end)
        self._controller.handle_click(*start, shift_held=True, goal_heading=goal_heading)
        self._drag_start_frame_px = None
        self._drag_current_frame_px = None

    def _heading_from_drag(self, start: tuple[int, int], end: tuple[int, int]) -> float | None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if math.sqrt(dx ** 2 + dy ** 2) < _DRAG_HEADING_THRESHOLD_PX:
            return None
        start_world = self._controller.map_coords.px_to_world(*start)
        end_world = self._controller.map_coords.px_to_world(*end)
        return math.atan2(end_world[1] - start_world[1], end_world[0] - start_world[0])

    def _frame_px_from_event(self, event: QMouseEvent) -> tuple[int, int] | None:
        frame_w, frame_h = self._controller.frame_size
        display_size = self._display.size()

        scale = min(display_size.width() / frame_w, display_size.height() / frame_h)
        scaled_w = int(frame_w * scale)
        scaled_h = int(frame_h * scale)

        offset_x = (display_size.width() - scaled_w) // 2
        offset_y = (display_size.height() - scaled_h) // 2

        label_height = 20
        click_x = event.position().x()
        click_y = event.position().y() - label_height

        frame_px_x = int((click_x - offset_x) / scale)
        frame_px_y = int((click_y - offset_y) / scale)

        if 0 <= frame_px_x < frame_w and 0 <= frame_px_y < frame_h:
            return frame_px_x, frame_px_y
        return None
