import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robot_commander.depth_processing.depth_capture import load

_LOGS_DIR = Path(__file__).parent / "logs"


def _latest_log_dir() -> Path:
    dirs = sorted(_LOGS_DIR.iterdir())
    if not dirs:
        raise FileNotFoundError(f"No run logs found in {_LOGS_DIR}")
    return dirs[-1]


def _load_captures(log_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    captures_dir = log_dir / "depth_captures"
    if not captures_dir.exists():
        raise FileNotFoundError(f"No depth_captures directory in {log_dir}")

    npz_files = sorted(captures_dir.glob("*.npz"), key=lambda p: float(p.stem))
    if not npz_files:
        raise FileNotFoundError(f"No depth captures found in {captures_dir}")

    timestamps = np.array([float(p.stem) for p in npz_files])
    timestamps -= timestamps[0]

    ultrasonic_values = np.array([load(p).ultrasonic_min for p in npz_files])
    return timestamps, ultrasonic_values


def plot_depth_captures(log_dir: Path) -> None:
    timestamps, ultrasonic_values = _load_captures(log_dir)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f"Depth captures — {log_dir.name}", fontsize=11)

    ax.plot(timestamps, ultrasonic_values * 100, color="steelblue", lw=1.2, marker="o", markersize=4, label="ultrasonic min")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ultrasonic min (cm)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    print(f"{len(timestamps)} depth captures over {timestamps[-1]:.1f}s")
    print(f"  ultrasonic min — mean: {ultrasonic_values.mean()*100:.1f} cm  "
          f"min: {ultrasonic_values.min()*100:.1f} cm  max: {ultrasonic_values.max()*100:.1f} cm")

    plt.tight_layout()
    output_path = log_dir / "plot_depth_captures.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ultrasonic min values from depth captures in a run log")
    parser.add_argument(
        "log_dir",
        type=Path,
        nargs="?",
        help="Path to the run log directory (default: most recent log)",
    )
    args = parser.parse_args()

    log_dir = args.log_dir or _latest_log_dir()
    print(f"Plotting depth captures from: {log_dir}")
    plot_depth_captures(log_dir)