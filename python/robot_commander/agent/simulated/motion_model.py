"""Pure functions for the simulated agent motion model."""

import math

V_MAX_M_S = 0.15
OMEGA_MAX_RAD_S = 3.5
A_MAX_M_S2 = 2.0
STEP_DURATION_S = 0.17
WAYPOINT_THRESHOLD_M = 0.05  # arrival radius for waypoints


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


def advance_velocity(
    vx: float,
    vy: float,
    desired_vx: float,
    desired_vy: float,
    dt: float,
) -> tuple[float, float]:
    max_dv = A_MAX_M_S2 * dt
    new_vx = vx + max(-max_dv, min(max_dv, desired_vx - vx))
    new_vy = vy + max(-max_dv, min(max_dv, desired_vy - vy))
    return new_vx, new_vy


def cardinal_direction(
    goal_x: float,
    goal_y: float,
    x: float,
    y: float,
    heading: float,
) -> tuple[float, float]:
    """Return the world-space unit vector for the cardinal direction relative
    to heading that best points toward the goal. The Adeept strafes rather
    than rotating its body, so movement is quantised to forward/back/left/right."""
    relative = normalize_angle(math.atan2(goal_y - y, goal_x - x) - heading)

    if -math.pi / 4 <= relative < math.pi / 4:
        angle = heading                  # forward
    elif math.pi / 4 <= relative < 3 * math.pi / 4:
        angle = heading + math.pi / 2   # left
    elif -3 * math.pi / 4 <= relative < -math.pi / 4:
        angle = heading - math.pi / 2   # right
    else:
        angle = heading + math.pi       # backward

    return math.cos(angle), math.sin(angle)