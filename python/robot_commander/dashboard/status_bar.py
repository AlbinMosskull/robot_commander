from datetime import datetime, timezone

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from robot_commander.remote_control.controller import LOCALIZATION_LOST_THRESHOLD, RemoteControl

_LOCALIZATION_SEARCHING_THRESHOLD = 5


class StatusBarWidget(QWidget):
    def __init__(self, controller: RemoteControl, parent=None):
        super().__init__(parent)
        self._controller = controller
        self.setFixedHeight(32)
        self.setStyleSheet("background-color: #111111; border-bottom: 1px solid #333333;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)

        connection_label = QLabel("Connection:")
        connection_label.setStyleSheet("color: #aaaaaa; font-family: monospace; font-size: 13px;")

        self._status_value = QLabel()
        self._status_value.setStyleSheet(
            "color: #00ff88; font-family: monospace; font-size: 13px; font-weight: bold;"
        )

        self._time_label = QLabel()
        self._time_label.setStyleSheet("color: #cccccc; font-family: monospace; font-size: 13px;")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        escape_plan_label = QLabel("Escape plan:")
        escape_plan_label.setStyleSheet("color: #aaaaaa; font-family: monospace; font-size: 13px;")

        self._escape_plan_value = QLabel()
        self._escape_plan_value.setStyleSheet(
            "color: #00ff88; font-family: monospace; font-size: 13px; font-weight: bold;"
        )

        self._jam_button = QPushButton("JAM LOCALIZATION")
        self._jam_button.setCheckable(True)
        self._jam_button.setStyleSheet(self._jam_button_style(False))
        self._jam_button.clicked.connect(self._on_jam_clicked)

        layout.addWidget(connection_label)
        layout.addWidget(self._status_value)
        layout.addSpacing(24)
        layout.addWidget(escape_plan_label)
        layout.addWidget(self._escape_plan_value)
        layout.addStretch()
        layout.addWidget(self._jam_button)
        layout.addSpacing(12)
        layout.addWidget(self._time_label)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_time)
        self._clock_timer.start(1000)
        self._update_time()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(200)
        self._update_status()

    def _update_status(self) -> None:
        miss_count = self._controller.localization_miss_count
        if miss_count == 0:
            text = "LOCALIZED"
            color = "#00ff88"
        elif miss_count < _LOCALIZATION_SEARCHING_THRESHOLD:
            text = f"SEARCHING ({miss_count})"
            color = "#ffaa00"
        else:
            text = f"NO SIGNAL ({miss_count})"
            color = "#ff4444"
        self._status_value.setText(text)
        self._status_value.setStyleSheet(
            f"color: {color}; font-family: monospace; font-size: 13px; font-weight: bold;"
        )

        age = self._controller.escape_plan_age_s
        if age is None:
            escape_text = "NONE"
            escape_color = "#ff4444"
        elif age < 2.0:
            escape_text = f"{age:.1f}s ago"
            escape_color = "#00ff88"
        elif age < 5.0:
            escape_text = f"{age:.1f}s ago"
            escape_color = "#ffaa00"
        else:
            escape_text = f"STALE ({age:.0f}s)"
            escape_color = "#ff4444"
        self._escape_plan_value.setText(escape_text)
        self._escape_plan_value.setStyleSheet(
            f"color: {escape_color}; font-family: monospace; font-size: 13px; font-weight: bold;"
        )

    def _on_jam_clicked(self) -> None:
        self._controller.toggle_localization_jam()
        jammed = self._controller.localization_jammed
        self._jam_button.setStyleSheet(self._jam_button_style(jammed))

    @staticmethod
    def _jam_button_style(active: bool) -> str:
        bg = "#ff4444" if active else "#333333"
        fg = "#ffffff" if active else "#aaaaaa"
        return (
            f"background-color: {bg}; color: {fg}; font-family: monospace; "
            f"font-size: 12px; font-weight: bold; border: 1px solid #555555; padding: 2px 8px;"
        )

    def _update_time(self) -> None:
        now = datetime.now(timezone.utc)
        self._time_label.setText(now.strftime("%H:%M:%S UTC"))
