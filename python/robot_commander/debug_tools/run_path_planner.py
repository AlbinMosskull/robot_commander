from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from robot_commander import OccupancyMap, WorldPosition2d, plan_path
from robot_commander.remote_control.obstacle_mapping import PlanPathFailure


def run_path_planner(failure: PlanPathFailure) -> list[tuple[float, float]] | None:
    height, width = failure.occ_grid.shape
    occ_map = OccupancyMap(
        width=width,
        height=height,
        resolution=failure.resolution,
        origin_x=failure.origin_x,
        origin_y=failure.origin_y,
    )
    occ_map.set_grid(failure.occ_grid.tolist())
    return [(p.x, p.y) for p in result] if (result := plan_path(occ_map, failure.start, failure.goal, failure.collision_margin)) else None


def run_path_planner_from_file(path: Path) -> list[tuple[float, float]] | None:
    data = np.load(path)
    failure = PlanPathFailure(
        start=WorldPosition2d(x=float(data["start"][0]), y=float(data["start"][1])),
        goal=WorldPosition2d(x=float(data["goal"][0]), y=float(data["goal"][1])),
        collision_margin=float(data["collision_margin"][0]),
        resolution=float(data["resolution"][0]),
        origin_x=float(data["origin"][0]),
        origin_y=float(data["origin"][1]),
        occ_grid=data["occ_grid"],
    )
    return run_path_planner(failure)


def write_obstacle_grid(failure: PlanPathFailure, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(failure.occ_grid, origin="lower", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)

    def world_to_pixel(wx: float, wy: float) -> tuple[float, float]:
        px = (wx - failure.origin_x) / failure.resolution
        py = (wy - failure.origin_y) / failure.resolution
        return px, py

    margin_radius_cells = failure.collision_margin / failure.resolution

    for world_pos, label, color in [
        (failure.start, "start", "blue"),
        (failure.goal, "goal", "orange"),
    ]:
        px, py = world_to_pixel(world_pos.x, world_pos.y)
        ax.plot(px, py, "o", color=color, markersize=8, label=label)
        circle = plt.Circle((px, py), margin_radius_cells, color=color, fill=False, linewidth=1.5, linestyle="--")
        ax.add_patch(circle)

    ax.legend(loc="upper right")
    ax.set_title("Occupancy grid (green=free, red=occupied)")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved occupancy grid visualization to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the path planner on a saved failure case")
    parser.add_argument("failure_file", type=Path, help="Path to the .npz file containing the failure case")
    parser.add_argument("--output_grid", type=Path, help="Optional path to save a visualization of the occupancy grid with start/goal")
    parser.add_argument("--override_goal", type=float, nargs=2, metavar=("X", "Y"), help="Override the goal position")
    parser.add_argument("--override_margin", type=float, help="Override the collision margin")
    args = parser.parse_args()

    data = np.load(args.failure_file)
    failure = PlanPathFailure(
        start=WorldPosition2d(x=float(data["start"][0]), y=float(data["start"][1])),
        goal=WorldPosition2d(x=float(args.override_goal[0]), y=float(args.override_goal[1])) if args.override_goal else WorldPosition2d(x=float(data["goal"][0]), y=float(data["goal"][1])),
        collision_margin=args.override_margin if args.override_margin is not None else float(data["collision_margin"][0]),
        resolution=float(data["resolution"][0]),
        origin_x=float(data["origin"][0]),
        origin_y=float(data["origin"][1]),
        occ_grid=data["occ_grid"],
    )

    if args.output_grid:
        write_obstacle_grid(failure, args.output_grid)

    path = run_path_planner(failure)
    if path is not None:
        print("Planned path:")
        for point in path:
            print(f"({point[0]:.2f}, {point[1]:.2f})")
    else:
        print("No valid path found")

