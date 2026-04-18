import argparse
import subprocess
import sys
from pathlib import Path

from robot_commander import config

_LOGS_DIR = Path(__file__).parent / "logs"
_REMOTE_PROJECT_PATH = "~/Code/robot_commander"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="pi", help="SSH user on the robot (default: pi)")
    args = parser.parse_args()

    agent_config = config.load().agent
    remote = f"{args.user}@{agent_config.host}:{_REMOTE_PROJECT_PATH}/python/robot_commander/debug_tools/logs/"
    result = subprocess.run(
        ["rsync", "-avz", "--progress", remote, str(_LOGS_DIR) + "/"],
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
