import grpc

from robot_commander.proto import agent_pb2, agent_pb2_grpc
from robot_commander import config as cfg


class AgentClient:
    def __init__(self, host: str = "localhost", port: int = cfg.load().agent.port):
        self._channel = grpc.insecure_channel(f"{host}:{port}")
        self._stub = agent_pb2_grpc.AgentControlStub(self._channel)

    def set_checkpoint(self, x: float, y: float) -> None:
        self._stub.SetCheckpoint(agent_pb2.Position(x=x, y=y))

    def set_path(self, waypoints: list[tuple[float, float]]) -> None:
        path = agent_pb2.Path(waypoints=[agent_pb2.Position(x=x, y=y) for x, y in waypoints])
        self._stub.SetPath(path)

    def observe_position(self, x: float, y: float, confidence: float) -> None:
        self._stub.ObservePosition(agent_pb2.PositionObservation(x=x, y=y, confidence=confidence))

    def stream_positions(self):
        for pos in self._stub.StreamPosition(agent_pb2.Empty()):
            yield pos.x, pos.y

    def stream_rays(self):
        for batch in self._stub.StreamRays(agent_pb2.Empty()):
            yield [(r.start_x, r.start_y, r.end_x, r.end_y, r.did_collide) for r in batch.rays]

    def close(self) -> None:
        self._channel.close()
