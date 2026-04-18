import argparse
import time

from robot_commander.agent.adeept.hardware import Ultra
from robot_commander.agent.adeept.hardware.Move import RaspClaws, _DEPTH_SENSOR_PAN_CHANNEL

_POLL_INTERVAL_S = 0.1
_SWEEP_STEP_DEGREES = 5
_SWEEP_STEP_INTERVAL_S = 0.05


def run_single():
    print("Ultrasonic readings (Ctrl+C to stop)\n")
    try:
        while True:
            distance_cm = Ultra.checkdist()
            print(f"{distance_cm:.1f} cm")
            time.sleep(_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        pass


def run_sweep(sweep_range_degrees: int):
    robot = RaspClaws()
    center_angle = robot.init_angles[_DEPTH_SENSOR_PAN_CHANNEL]
    min_angle = center_angle - sweep_range_degrees
    max_angle = center_angle + sweep_range_degrees
    angle = center_angle
    direction = 1

    print(f"Sweeping depth sensor ±{sweep_range_degrees}° (Ctrl+C to stop)\n")
    try:
        while True:
            robot.set_servo_angle(_DEPTH_SENSOR_PAN_CHANNEL, angle)
            distance_cm = Ultra.checkdist()
            print(f"{angle:6.1f}°  {distance_cm:.1f} cm")
            time.sleep(_SWEEP_STEP_INTERVAL_S)

            angle += direction * _SWEEP_STEP_DEGREES
            if angle >= max_angle:
                angle = max_angle
                direction = -1
            elif angle <= min_angle:
                angle = min_angle
                direction = 1
    except KeyboardInterrupt:
        robot.set_servo_angle(_DEPTH_SENSOR_PAN_CHANNEL, center_angle)
        robot.release_servo(_DEPTH_SENSOR_PAN_CHANNEL)
        robot.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-range", type=int, default=45, metavar="DEGREES")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args.sweep_range)
    else:
        run_single()


if __name__ == "__main__":
    main()
