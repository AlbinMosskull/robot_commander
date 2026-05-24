import argparse

from robot_commander import config as cfg
from robot_commander.remote_control.agent_client import AgentClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint = subparsers.add_parser("checkpoint", help="Set a single waypoint")
    checkpoint.add_argument("x", type=float)
    checkpoint.add_argument("y", type=float)

    path = subparsers.add_parser("path", help="Set a multi-waypoint path (each waypoint as x,y)")
    path.add_argument("waypoints", nargs="+", metavar="x,y")

    subparsers.add_parser("status", help="Print current agent position and exit")
    subparsers.add_parser("scout", help="Send scout command")

    return parser.parse_args()


def _parse_waypoint(raw: str) -> tuple[float, float]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Waypoint must be x,y — got: {raw!r}")
    return float(parts[0]), float(parts[1])


def main() -> None:
    args = _parse_args()
    connection = cfg.load().connection
    host = connection.simulated_host if args.simulate else connection.host
    client = AgentClient(host=host)

    try:
        if args.command == "checkpoint":
            client.set_checkpoint(args.x, args.y)
            print(f"checkpoint set  x={args.x}  y={args.y}")
        elif args.command == "path":
            waypoints = [_parse_waypoint(w) for w in args.waypoints]
            client.set_path(waypoints)
            formatted = "  ".join(f"({x},{y})" for x, y in waypoints)
            print(f"path set  waypoints={formatted}")
        elif args.command == "status":
            x, y, heading = next(iter(client.stream_positions()))
            print(f"pos  x={x:.3f}  y={y:.3f}  heading={heading:.3f}")
        elif args.command == "scout":
            client.scout()
            print("scout command sent")
    finally:
        client.close()


if __name__ == "__main__":
    main()
