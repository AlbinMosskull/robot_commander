import sys
import tty
import termios

from robot_commander.remote_control.agent_client import AgentClient


COMMANDS = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    " ": "stand",
}

HELP_TEXT = """
Remote manual control — Adeept RaspClaws (via gRPC)
  w  forward
  s  backward
  a  left
  d  right
  [space]  stop
  q  quit

Each keypress moves the robot for 0.5s then stops.
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
    client = AgentClient()

    while True:
        key = read_single_key()
        if key == "q":
            client.run_command("stand", 0.0)
            client.close()
            print("\nExiting.")
            break
        command = COMMANDS.get(key)
        if command:
            print(f"  {command}")
            client.run_command(command, 0.5)


if __name__ == "__main__":
    main()
