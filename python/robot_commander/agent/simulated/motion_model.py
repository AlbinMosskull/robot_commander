"""Pure functions for the simulated agent motion model."""

import math

V_MAX_M_S = 0.15
OMEGA_MAX_RAD_S = 3.5
A_MAX_M_S2 = 2.0
WAYPOINT_THRESHOLD_M = 0.05
HEADING_ALIGNMENT_RAD = math.pi / 6


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def advance_heading(
    heading: float,
    goal_x: float,
    goal_y: float,
    x: float,
    y: float,
    dt: float,
) -> float:
    desired = math.atan2(goal_y - y, goal_x - x)
    error = normalize_angle(desired - heading)
    max_turn = OMEGA_MAX_RAD_S * dt
    return heading + max(-max_turn, min(max_turn, error))


def advance_speed(speed: float, desired_speed: float, dt: float) -> float:
    max_dv = A_MAX_M_S2 * dt
    return speed + max(-max_dv, min(max_dv, desired_speed - speed))
