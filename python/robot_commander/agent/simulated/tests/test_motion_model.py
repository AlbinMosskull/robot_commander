import math

from robot_commander.agent.simulated.motion_model import (
    normalize_angle,
    advance_heading,
    advance_speed,
    OMEGA_MAX_RAD_S,
    A_MAX_M_S2,
)


def test_normalize_angle_wraps_to_minus_pi_pi():
    assert math.isclose(normalize_angle(math.pi + 0.5), -math.pi + 0.5, abs_tol=1e-9)
    assert math.isclose(normalize_angle(-math.pi - 0.5), math.pi - 0.5, abs_tol=1e-9)


def test_advance_heading_clamps_to_max_turn():
    dt = 0.1
    new_heading = advance_heading(0.0, goal_x=0.0, goal_y=1.0, x=0.0, y=0.0, dt=dt)
    assert math.isclose(new_heading, OMEGA_MAX_RAD_S * dt, abs_tol=1e-9)


def test_advance_speed_clamps_acceleration():
    dt = 0.1
    new_speed = advance_speed(0.0, desired_speed=10.0, dt=dt)
    assert math.isclose(new_speed, A_MAX_M_S2 * dt, abs_tol=1e-9)


def test_advance_speed_decelerates():
    dt = 0.1
    new_speed = advance_speed(1.0, desired_speed=0.0, dt=dt)
    assert math.isclose(new_speed, 1.0 - A_MAX_M_S2 * dt, abs_tol=1e-9)