import threading

import numpy as np

from robot_commander.localization.world_localizer import WorldLocalizer, WorldPose
from robot_commander.remote_control.agent_client import AgentClient


class SimulatedLocalizer(WorldLocalizer):
    def __init__(self, client: AgentClient):
        self._client = client
        self._latest: tuple[float, float, float] | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._stream, daemon=True)
        self._thread.start()

    def localize(self, frame: np.ndarray) -> WorldPose | None:
        with self._lock:
            pos = self._latest
        if pos is None:
            return None
        x, y, heading = pos
        noisy_x, noisy_y = np.random.normal([x, y], scale=0.02)
        self._client.observe_position(float(noisy_x), float(noisy_y), heading, confidence=1.0)
        return WorldPose(x, y, heading)

    def _stream(self) -> None:
        try:
            for x, y, heading in self._client.stream_positions():
                with self._lock:
                    self._latest = (x, y, heading)
        except Exception:
            pass
