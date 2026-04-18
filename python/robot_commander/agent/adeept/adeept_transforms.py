import numpy as np

ROBOT_T_CAMERA: np.ndarray = np.eye(4, dtype=np.float64)

CAMERA_T_SENSOR_CENTER: np.ndarray = np.array([
    [1.0, 0.0, 0.0,  0.00],
    [0.0, 1.0, 0.0, -0.10],
    [0.0, 0.0, 1.0, -0.05],
    [0.0, 0.0, 0.0,  1.00],
], dtype=np.float64)

# Increasing servo angle rotates the sensor left (matches 'lookleft' on channel 12).
# Flip to +1 if physically reversed.
_SENSOR_PAN_DIRECTION = -1


def camera_T_sensor(servo_angle_offset_rad: float) -> np.ndarray:
    """Transform from sensor frame to camera frame at a given servo angle offset from centre."""
    c = np.cos(servo_angle_offset_rad)
    s = np.sin(servo_angle_offset_rad) * _SENSOR_PAN_DIRECTION
    rotation_Y = np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float64)
    return CAMERA_T_SENSOR_CENTER @ rotation_Y


def robot_T_sensor(servo_angle_offset_rad: float) -> np.ndarray:
    """Transform from sensor frame to robot frame at a given servo angle offset from centre."""
    return ROBOT_T_CAMERA @ camera_T_sensor(servo_angle_offset_rad)
