import os
import sys

# opencv-contrib-python sets QT_QPA_PLATFORM_PLUGIN_PATH to its own bundled Qt
# plugins at import time, which prevents PyQt6 from finding its own plugins.
# Import cv2 first to trigger that side-effect, then clear the variable so
# PyQt6 resolves its platform plugin correctly.
import cv2  # noqa: F401
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt6.QtWidgets import QApplication

from robot_commander.dashboard.window import DashboardWindow


def main():
    app = QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
