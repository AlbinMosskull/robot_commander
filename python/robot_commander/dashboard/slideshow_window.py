from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QMainWindow, QProgressBar, QVBoxLayout, QWidget


class SlideshowWindow(QMainWindow):
    finished = pyqtSignal()

    def __init__(self, images: list[Path], duration_ms: int = 1000):
        super().__init__()
        self.setWindowTitle("Robot Commander")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #1a1a1a; color: #e0e0e0; font-family: monospace;")

        self._images = images
        self._index = 0

        root_widget = QWidget()
        self.setCentralWidget(root_widget)

        layout = QVBoxLayout(root_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #0d0d0d;")
        layout.addWidget(self._image_label, stretch=1)

        footer = QWidget()
        footer.setFixedHeight(48)
        footer.setStyleSheet("background-color: #111111; padding: 6px 12px;")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(12, 6, 12, 6)
        footer_layout.setSpacing(4)

        status_label = QLabel("Building overhead map...")
        status_label.setStyleSheet("color: #888888; font-size: 11px;")
        footer_layout.addWidget(status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, len(self._images))
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background-color: #2a2a2a; border: none; border-radius: 2px; }"
            "QProgressBar::chunk { background-color: #4a9eff; border-radius: 2px; }"
        )
        footer_layout.addWidget(self._progress_bar)

        layout.addWidget(footer)

        self._duration_ms = duration_ms
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def _show_current(self) -> None:
        path = self._images[self._index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._advance()
            return
        self._image_label.setPixmap(
            pixmap.scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._progress_bar.setValue(self._index + 1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._show_current()
        self._timer.start(self._duration_ms)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._images:
            self._show_current()

    def _advance(self) -> None:
        self._index += 1
        if self._index >= len(self._images):
            self._timer.stop()
            self.finished.emit()
            self.close()
        else:
            self._show_current()
