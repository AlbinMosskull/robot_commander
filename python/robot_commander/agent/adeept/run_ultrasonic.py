import time

from robot_commander.agent.adeept.hardware import Ultra

_POLL_INTERVAL_S = 0.1


def main():
    print("Ultrasonic readings (Ctrl+C to stop)\n")
    try:
        while True:
            distance_cm = Ultra.checkdist()
            print(f"{distance_cm:.1f} cm")
            time.sleep(_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
