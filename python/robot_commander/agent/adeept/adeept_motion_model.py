"""Pure functions for the Adeept robot motion model."""

import math

from robot_commander.config import load as load_config

V_FORWARD_M_S = load_config().agent.v_forward_m_s
HEADING_ALIGNMENT_RAD = math.pi / 6
WAYPOINT_THRESHOLD_M = 0.01


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def direction_command(heading_error: float) -> str:
    """Return the robot command for a given heading error (radians)."""
    if abs(heading_error) < HEADING_ALIGNMENT_RAD:
        return "forward"
    return "left" if heading_error > 0 else "right"


def predict_displacement(command: str, heading: float, dt: float) -> tuple[float, float]:
    """Return expected (dx, dy) displacement for a given command."""
    if command == "forward":
        return V_FORWARD_M_S * math.cos(heading) * dt, V_FORWARD_M_S * math.sin(heading) * dt
    return 0.0, 0.0
