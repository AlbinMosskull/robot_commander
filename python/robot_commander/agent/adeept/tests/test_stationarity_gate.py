import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

_PI_MODULES = [
    "picamera2",
    "robot_commander.agent.adeept.hardware",
    "robot_commander.agent.adeept.hardware.Move",
    "robot_commander.agent.adeept.hardware.Ultra",
    "robot_commander.agent.adeept.hardware.mpu6050_gyro",
]

for _mod in _PI_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

import robot_commander.agent.adeept.adeept_agent as _agent_module  # noqa: E402
from robot_commander.agent.adeept.adeept_agent import (  # noqa: E402
    AdeeptAgent,
    _ULTRA_HIT_THRESHOLD_CM,
    _STAND_SETTLE_S,
)


def _make_agent(command: str, last_motion_seconds_ago: float) -> AdeeptAgent:
    with patch.object(AdeeptAgent, "__init__", lambda self, *a, **kw: None):
        agent = AdeeptAgent.__new__(AdeeptAgent)
    agent._raw_sensor = False
    agent._current_command = command
    agent._last_motion_time = time.time() - last_motion_seconds_ago
    agent._lock = threading.Lock()
    return agent


def test_returns_none_when_moving():
    agent = _make_agent(command="forward", last_motion_seconds_ago=10.0)
    assert agent.GetUltrasonicMin() is None


def test_returns_none_when_turning():
    agent = _make_agent(command="left", last_motion_seconds_ago=10.0)
    assert agent.GetUltrasonicMin() is None


def test_returns_none_when_not_yet_settled():
    agent = _make_agent(command="stand", last_motion_seconds_ago=0.0)
    assert agent.GetUltrasonicMin() is None


def test_proceeds_after_settle_time():
    agent = _make_agent(command="stand", last_motion_seconds_ago=_STAND_SETTLE_S + 0.1)
    _agent_module.Ultra.checkdist.return_value = 50.0
    result = agent.GetUltrasonicMin()
    assert result == pytest.approx(0.5)


def test_returns_none_when_ultrasonic_reads_too_far():
    agent = _make_agent(command="stand", last_motion_seconds_ago=_STAND_SETTLE_S + 0.1)
    _agent_module.Ultra.checkdist.return_value = _ULTRA_HIT_THRESHOLD_CM + 1
    result = agent.GetUltrasonicMin()
    assert result is None
