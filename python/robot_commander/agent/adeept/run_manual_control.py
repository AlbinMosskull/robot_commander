import sys
import tty
import termios
from robot_commander.agent.adeept.hardware.Move import RaspClaws


COMMANDS = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    " ": "stand",
}

_IDLE_SERVO_CHANNELS = [12, 13, 14, 15]

HELP_TEXT = """
Manual control — Adeept RaspClaws
  w  forward
  s  backward
  a  left
  d  right
  [space]  stop
  q  quit
"""


def read_single_key() -> str:
    file_descriptor = sys.stdin.fileno()
    old_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setraw(file_descriptor)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, old_settings)


def main():
    print(HELP_TEXT)

    robot = RaspClaws()
    robot.daemon = True
    robot.start()
    for ch in _IDLE_SERVO_CHANNELS:
        robot.release_servo(ch)

    while True:
        key = read_single_key()
        if key == "q":
            robot.command_input("stand")
            print("\nExiting.")
            break
        command = COMMANDS.get(key)
        if command:
            print(f"  {command}")
            robot.command_input(command)


if __name__ == "__main__":
    main()
