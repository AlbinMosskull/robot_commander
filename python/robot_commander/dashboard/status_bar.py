from datetime import datetime, timezone

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget


class StatusBarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setStyleSheet("background-color: #111111; border-bottom: 1px solid #333333;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)

        connection_label = QLabel("Connection:")
        connection_label.setStyleSheet("color: #aaaaaa; font-family: monospace; font-size: 13px;")

        self._status_value = QLabel("ALIVE")
        self._status_value.setStyleSheet(
            "color: #00ff88; font-family: monospace; font-size: 13px; font-weight: bold;"
        )

        self._time_label = QLabel()
        self._time_label.setStyleSheet("color: #cccccc; font-family: monospace; font-size: 13px;")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(connection_label)
        layout.addWidget(self._status_value)
        layout.addStretch()
        layout.addWidget(self._time_label)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_time)
        self._timer.start(1000)
        self._update_time()

    def _update_time(self) -> None:
        now = datetime.now(timezone.utc)
        self._time_label.setText(now.strftime("%H:%M:%S UTC"))
