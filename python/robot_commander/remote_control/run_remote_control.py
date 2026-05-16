import argparse
import os
import sys
from pathlib import Path

# opencv-contrib-python sets QT_QPA_PLATFORM_PLUGIN_PATH to its own bundled Qt
# plugins at import time, which prevents PyQt6 from finding its own plugins.
# Import cv2 first to trigger that side-effect, then clear the variable so
# PyQt6 resolves its platform plugin correctly.
import cv2  # noqa: F401
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt6.QtWidgets import QApplication

from robot_commander import config as cfg
from robot_commander.dashboard.slideshow_window import SlideshowWindow
from robot_commander.dashboard.window import DashboardWindow

_MAP_BUILD_PROGRESS_DIR = Path("plots/debug")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-sim-agent", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    connection = cfg.load().connection
    agent_host = connection.simulated_host if args.use_sim_agent else None

    app = QApplication(sys.argv)
    dashboard = DashboardWindow(show_escape_plan=True, agent_host=agent_host)

    map_build_images = (
        sorted(
            (p for p in _MAP_BUILD_PROGRESS_DIR.glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES),
            key=lambda p: p.name,
        )
        if _MAP_BUILD_PROGRESS_DIR.exists()
        else []
    )

    if map_build_images:
        slideshow = SlideshowWindow(map_build_images)
        slideshow.finished.connect(dashboard.show)
        slideshow.show()
    else:
        dashboard.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
