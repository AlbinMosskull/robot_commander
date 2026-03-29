import grpc

from robot_commander.proto import agent_pb2, agent_pb2_grpc
from robot_commander import config as cfg


class AgentClient:
    def __init__(self, host: str = "localhost", port: int = cfg.load().agent.port):
        self._channel = grpc.insecure_channel(f"{host}:{port}")
        self._stub = agent_pb2_grpc.AgentControlStub(self._channel)

    def set_checkpoint(self, x: float, y: float) -> None:
        self._stub.SetCheckpoint(agent_pb2.Position(x=x, y=y))

    def stream_positions(self):
        for pos in self._stub.StreamPosition(agent_pb2.Empty()):
            yield pos.x, pos.y

    def stream_rays(self):
        for batch in self._stub.StreamRays(agent_pb2.Empty()):
            yield [(r.start_x, r.start_y, r.end_x, r.end_y, r.did_collide) for r in batch.rays]

    def close(self) -> None:
        self._channel.close()
