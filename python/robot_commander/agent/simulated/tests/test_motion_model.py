import math

from robot_commander.agent.simulated.motion_model import (
    normalize_angle,
    advance_heading,
    advance_velocity,
    cardinal_direction,
    OMEGA_MAX_RAD_S,
    A_MAX_M_S2,
)


def test_normalize_angle_wraps_to_minus_pi_pi():
    assert math.isclose(normalize_angle(math.pi + 0.5), -math.pi + 0.5, abs_tol=1e-9)
    assert math.isclose(normalize_angle(-math.pi - 0.5), math.pi - 0.5, abs_tol=1e-9)


def test_advance_heading_clamps_to_max_turn():
    dt = 0.1
    # Goal is 90° left of current heading — turn should be capped
    new_heading = advance_heading(0.0, goal_x=0.0, goal_y=1.0, x=0.0, y=0.0, dt=dt)
    assert math.isclose(new_heading, OMEGA_MAX_RAD_S * dt, abs_tol=1e-9)


def test_advance_velocity_clamps_acceleration():
    dt = 0.1
    new_vx, new_vy = advance_velocity(0.0, 0.0, desired_vx=10.0, desired_vy=10.0, dt=dt)
    assert math.isclose(new_vx, A_MAX_M_S2 * dt, abs_tol=1e-9)
    assert math.isclose(new_vy, A_MAX_M_S2 * dt, abs_tol=1e-9)


def test_cardinal_direction_selects_correct_quadrant():
    # Heading east (0 rad), goal to the south → right
    dx, dy = cardinal_direction(goal_x=0.0, goal_y=-1.0, x=0.0, y=0.0, heading=0.0)
    assert math.isclose(dx, 0.0, abs_tol=1e-6)
    assert math.isclose(dy, -1.0, abs_tol=1e-6)

    # Heading east, goal to the west → backward
    dx, dy = cardinal_direction(goal_x=-1.0, goal_y=0.0, x=0.0, y=0.0, heading=0.0)
    assert math.isclose(dx, -1.0, abs_tol=1e-6)
    assert math.isclose(dy, 0.0, abs_tol=1e-6)
