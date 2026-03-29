import time
import threading
from concurrent import futures

import grpc

from robot_commander.agent.agent import Agent
from robot_commander.proto import agent_pb2, agent_pb2_grpc
from robot_commander import config as cfg

_TICK_HZ = 10


class AgentControlServicer(agent_pb2_grpc.AgentControlServicer):
    def __init__(self):
        self._agent = Agent(x=0.0, y=0.0, v=0.05)
        self._lock = threading.Lock()
        self._goal: tuple[float, float] | None = None

    def SetCheckpoint(self, request, context):
        with self._lock:
            self._goal = (request.x, request.y)
        return agent_pb2.Empty()

    def StreamPosition(self, request, context):
        while context.is_active():
            with self._lock:
                if self._goal is not None:
                    self._agent.move(*self._goal)
                x, y = self._agent.x, self._agent.y
            yield agent_pb2.Position(x=x, y=y)
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
