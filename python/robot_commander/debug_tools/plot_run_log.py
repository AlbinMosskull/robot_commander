import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

_LOGS_DIR = Path(__file__).parent / "logs"


def _latest_log_dir() -> Path:
    dirs = sorted(_LOGS_DIR.iterdir())
    if not dirs:
        raise FileNotFoundError(f"No run logs found in {_LOGS_DIR}")
    return dirs[-1]


def _load_obs(log_dir: Path, t0: float) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    obs_path = log_dir / "observations.csv"
    if obs_path.stat().st_size == 0:
        return None, None
    obs = pd.read_csv(obs_path)
    obs["t"] = obs["timestamp_s"] - t0
    return obs[~obs["heading_rejected"]], obs[obs["heading_rejected"]]


def _plot_position(ticks: pd.DataFrame, accepted_obs: pd.DataFrame | None, log_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(f"Position debug: {log_dir.name}", fontsize=11)

    ax.plot(ticks["pos_x_m"], ticks["pos_y_m"], color="steelblue", lw=1.2, label="Kalman position")
    ax.scatter(ticks["pos_x_m"].iloc[0], ticks["pos_y_m"].iloc[0], color="green", s=60, zorder=4, label="start")
    ax.scatter(ticks["pos_x_m"].iloc[-1], ticks["pos_y_m"].iloc[-1], color="red", s=60, zorder=4, label="end")

    if accepted_obs is not None and not accepted_obs.empty:
        ax.scatter(accepted_obs["observed_x_m"], accepted_obs["observed_y_m"],
                   s=20, color="limegreen", zorder=3, label="localizer position")

    has_target = ticks["target_x_m"].notna()
    if has_target.any():
        ax.scatter(ticks.loc[has_target, "target_x_m"], ticks.loc[has_target, "target_y_m"],
                   s=10, color="magenta", alpha=0.4, zorder=2, label="target")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = log_dir / "plot_position.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def _plot_heading(ticks: pd.DataFrame, accepted_obs: pd.DataFrame | None, rejected_obs: pd.DataFrame | None, log_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Heading debug: {log_dir.name}", fontsize=11)

    ax.plot(ticks["t"], np.degrees(ticks["heading_rad"]), color="steelblue", lw=1.2, label="Kalman heading")

    has_gyro = ticks["gyro_heading_rad"].notna() if "gyro_heading_rad" in ticks.columns else pd.Series(False, index=ticks.index)
    if has_gyro.any():
        ax.plot(ticks.loc[has_gyro, "t"], np.degrees(ticks.loc[has_gyro, "gyro_heading_rad"]),
                color="mediumpurple", lw=1.0, linestyle="--", label="gyro heading")

    has_target = ticks["heading_error_rad"].notna()
    if has_target.any():
        raw = ticks.loc[has_target, "heading_rad"] + ticks.loc[has_target, "heading_error_rad"]
        target_heading = (raw + np.pi) % (2 * np.pi) - np.pi
        ax.plot(ticks.loc[has_target, "t"], np.degrees(target_heading),
                color="darkorange", lw=1.0, linestyle=":", label="target heading")

    if accepted_obs is not None and not accepted_obs.empty:
        ax.scatter(accepted_obs["t"], np.degrees(accepted_obs["observed_heading_rad"]),
                   s=14, color="orange", zorder=3, label="localizer heading (accepted)")
    if rejected_obs is not None and not rejected_obs.empty:
        ax.scatter(rejected_obs["t"], np.degrees(rejected_obs["observed_heading_rad"]),
                   s=14, color="red", marker="x", zorder=3, label="localizer heading (rejected)")

    ax.set_ylabel("Heading (°)")
    ax.set_xlabel("Time (s)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f°"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = log_dir / "plot_heading.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def plot_run_log(log_dir: Path, heading: bool = False, position: bool = False) -> None:
    ticks = pd.read_csv(log_dir / "ticks.csv")
    t0 = ticks["timestamp_s"].iloc[0]
    ticks["t"] = ticks["timestamp_s"] - t0

    accepted_obs, rejected_obs = _load_obs(log_dir, t0)

    if heading:
        _plot_heading(ticks, accepted_obs, rejected_obs, log_dir)
        return

    if position:
        _plot_position(ticks, accepted_obs, log_dir)
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Run log: {log_dir.name}", fontsize=11)

    ax = axes[0]
    ax.plot(ticks["t"], np.degrees(ticks["heading_rad"]), color="steelblue", lw=1.2, label="filter heading")
    if accepted_obs is not None and not accepted_obs.empty:
        ax.scatter(accepted_obs["t"], np.degrees(accepted_obs["observed_heading_rad"]),
                   s=12, color="orange", zorder=3, label="camera observation (accepted)")
    if rejected_obs is not None and not rejected_obs.empty:
        ax.scatter(rejected_obs["t"], np.degrees(rejected_obs["observed_heading_rad"]),
                   s=12, color="red", marker="x", zorder=3, label="camera observation (rejected)")
    ax.set_ylabel("Heading (°)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f°"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ticks["t"], ticks["heading_variance"], color="purple", lw=1.0, label="heading variance")
    if accepted_obs is not None and not accepted_obs.empty:
        ax.scatter(accepted_obs["t"], accepted_obs["heading_innovation_rad"].abs(),
                   s=10, color="orange", zorder=3, label="|innovation| (accepted)")
    ax.axhline(np.pi / 4, color="red", lw=0.8, linestyle="--", label="rejection threshold (45°)")
    ax.set_ylabel("Variance / |innovation| (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    has_target = ticks["heading_error_rad"].notna()
    ax.plot(ticks.loc[has_target, "t"], np.degrees(ticks.loc[has_target, "heading_error_rad"]),
            color="darkorange", lw=1.0, label="heading error to target")
    ax.axhline(30, color="grey", lw=0.8, linestyle="--", label="±alignment threshold (30°)")
    ax.axhline(-30, color="grey", lw=0.8, linestyle="--")
    ax.set_ylabel("Heading error (°)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f°"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(ticks.loc[has_target, "t"], ticks.loc[has_target, "distance_to_target_m"],
            color="teal", lw=1.0, label="distance to target (m)")
    ax.set_ylabel("Distance (m)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    command_colors = {"forward": "green", "left": "dodgerblue", "right": "tomato", "stand": "grey"}
    for _, row in ticks.iterrows():
        color = command_colors.get(row["command"], "black")
        ax.axvline(row["t"], color=color, alpha=0.08, lw=1.0)

    command_handles = [
        plt.Line2D([0], [0], color=c, lw=4, alpha=0.5, label=cmd)
        for cmd, c in command_colors.items()
    ]
    ax.legend(handles=command_handles + ax.get_legend_handles_labels()[0],
              fontsize=8, loc="upper right")

    plt.tight_layout()
    output_path = log_dir / "plot.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a recorded agent run log")
    parser.add_argument(
        "log_dir",
        type=Path,
        nargs="?",
        help="Path to the run log directory (default: most recent log)",
    )
    parser.add_argument(
        "--heading",
        action="store_true",
        help="Show heading debug plot (gyro, Kalman, localizer, target)",
    )
    parser.add_argument(
        "--position",
        action="store_true",
        help="Show bird's-eye position plot",
    )
    args = parser.parse_args()

    log_dir = args.log_dir or _latest_log_dir()
    print(f"Plotting: {log_dir}")
    plot_run_log(log_dir, heading=args.heading, position=args.position)
