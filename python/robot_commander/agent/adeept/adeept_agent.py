import math
import threading
import time

import numpy as np

from robot_commander.agent.abstract_agent import AbstractAgent
from robot_commander.agent.types import RangeReading
from robot_commander.filtering.kalman_filter import KalmanFilter
from robot_commander.hardware.adeept.Move import RaspClaws
from robot_commander.hardware.adeept import Ultra

_TICK_HZ = 10
_DT = 1.0 / _TICK_HZ
_ULTRA_HIT_THRESHOLD_CM = 190.0  # treat as no-hit near the 200cm max range



