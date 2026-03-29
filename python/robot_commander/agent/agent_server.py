import math
import time
import threading
from concurrent import futures

import grpc

from robot_commander.agent.agent import Agent
from robot_commander.agent.ray_caster import cast_rays
from robot_commander.proto import agent_pb2, agent_pb2_grpc
from robot_commander import config as cfg

_TICK_HZ = 10


class AgentControlServicer(agent_pb2_grpc.AgentControlServicer):
    def __init__(self):
        self._agent = Agent(x=0.0, y=0.0, v=0.05)
        self._lock = threading.Lock()
        self._goal: tuple[float, float] | None = None
        self._heading: float = math.pi / 2
        self._rays: list[tuple[float, float, float, float]] = []

        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def _tick_loop(self):
        while True:
            with self._lock:
                prev_x, prev_y = self._agent.x, self._agent.y
                if self._goal is not None:
                    self._agent.move(*self._goal)
                x, y = self._agent.x, self._agent.y
                dx, dy = x - prev_x, y - prev_y
                if dx * dx + dy * dy > 1e-8:
                    self._heading = math.atan2(dy, dx)
                self._rays = cast_rays(x, y, self._heading)
            time.sleep(1 / _TICK_HZ)

    def SetCheckpoint(self, request, context):
        with self._lock:
            self._goal = (request.x, request.y)
        return agent_pb2.Empty()

    def StreamPosition(self, request, context):
        while context.is_active():
            with self._lock:
                x, y = self._agent.x, self._agent.y
            yield agent_pb2.Position(x=x, y=y)
            time.sleep(1 / _TICK_HZ)

    def StreamRays(self, request, context):
        while context.is_active():
            with self._lock:
                rays = self._rays
            yield agent_pb2.RayBatch(rays=[
                agent_pb2.Ray(start_x=sx, start_y=sy, end_x=ex, end_y=ey)
                for sx, sy, ex, ey in rays
            ])
            time.sleep(1 / _TICK_HZ)


def main():
    port = cfg.load().agent.port
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    agent_pb2_grpc.add_AgentControlServicer_to_server(AgentControlServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Agent server listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
