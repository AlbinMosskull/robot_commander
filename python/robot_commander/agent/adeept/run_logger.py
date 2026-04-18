import csv
import time
from pathlib import Path

_LOGS_DIR = Path(__file__).parent.parent.parent / "debug_tools" / "logs"


class RunLogger:
    def __init__(self):
        run_dir = _LOGS_DIR / time.strftime("%Y%m%dT%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        self._tick_file = open(run_dir / "ticks.csv", "w", newline="")
        self._obs_file = open(run_dir / "observations.csv", "w", newline="")

        self._tick_writer = csv.writer(self._tick_file)
        self._obs_writer = csv.writer(self._obs_file)

        self._tick_writer.writerow([
            "timestamp_s", "heading_rad", "heading_variance",
            "pos_x_m", "pos_y_m", "command",
            "heading_error_rad", "distance_to_target_m",
            "target_x_m", "target_y_m",
            "gyro_heading_rad",
        ])
        self._obs_writer.writerow([
            "timestamp_s",
            "observed_heading_rad", "heading_innovation_rad", "heading_rejected",
            "observed_x_m", "observed_y_m",
            "pos_innovation_x_m", "pos_innovation_y_m",
        ])

    def log_tick(
        self,
        heading: float,
        heading_variance: float,
        pos_x: float,
        pos_y: float,
        command: str,
        heading_error: float | None,
        distance_to_target: float | None,
        target_x: float | None,
        target_y: float | None,
        gyro_heading: float | None,
    ) -> None:
        self._tick_writer.writerow([
            time.monotonic(), heading, heading_variance,
            pos_x, pos_y, command,
            heading_error, distance_to_target,
            target_x, target_y,
            gyro_heading,
        ])

    def log_observation(
        self,
        observed_heading: float,
        heading_innovation: float | None,
        observed_x: float,
        observed_y: float,
        pos_innovation_x: float,
        pos_innovation_y: float,
    ) -> None:
        self._obs_writer.writerow([
            time.monotonic(),
            observed_heading, heading_innovation, heading_innovation is None,
            observed_x, observed_y,
            pos_innovation_x, pos_innovation_y,
        ])

    def close(self) -> None:
        self._tick_file.close()
        self._obs_file.close()
