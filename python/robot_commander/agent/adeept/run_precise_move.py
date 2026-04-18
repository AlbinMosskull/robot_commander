"""Control the robot to move a precise distance or turn a precise angle."""

import argparse
import math

from robot_commander.agent.adeept.adeept_motion_model import V_FORWARD_M_S, OMEGA_MAX_RAD_S
from robot_commander.remote_control.agent_client import AgentClient


def duration_for_distance_cm(distance_cm: float) -> float:
    return abs(distance_cm) / 100.0 / V_FORWARD_M_S


def duration_for_degrees(degrees: float) -> float:
    return abs(math.radians(degrees)) / OMEGA_MAX_RAD_S


def move(distance_cm: float) -> None:
    command = "forward" if distance_cm > 0 else "backward"
    duration = duration_for_distance_cm(distance_cm)
    print(f"{command} {abs(distance_cm):.1f} cm ({duration:.2f}s)")
    client = AgentClient()
    client.run_command(command, duration)
    client.close()


def turn(degrees: float) -> None:
    command = "left" if degrees > 0 else "right"
    duration = duration_for_degrees(degrees)
    print(f"{command} {abs(degrees):.1f}° ({duration:.2f}s)")
    client = AgentClient()
    client.run_command(command, duration)
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precise robot movement control")
    subparsers = parser.add_subparsers(dest="action", required=True)

    move_parser = subparsers.add_parser("move", help="Move forward or backward")
    move_parser.add_argument(
        "distance_cm",
        type=float,
        help="Distance in centimetres (positive = forward, negative = backward)",
    )

    turn_parser = subparsers.add_parser("turn", help="Turn left or right")
    turn_parser.add_argument(
        "degrees",
        type=float,
        help="Angle in degrees (positive = left, negative = right)",
    )

    args = parser.parse_args()

    if args.action == "move":
        move(args.distance_cm)
    elif args.action == "turn":
        turn(args.degrees)


if __name__ == "__main__":
    main()
