import grpc

from robot_commander.proto import agent_pb2, agent_pb2_grpc
from robot_commander import config as cfg


class AgentClient:
    def __init__(self, host: str = cfg.load().agent.host, port: int = cfg.load().agent.port):
        self._channel = grpc.insecure_channel(f"{host}:{port}")
        self._stub = agent_pb2_grpc.AgentControlStub(self._channel)

    def set_checkpoint(self, x: float, y: float) -> None:
        self._stub.SetCheckpoint(agent_pb2.Position(x=x, y=y))

    def set_path(self, waypoints: list[tuple[float, float]], final_heading: float | None = None) -> None:
        kwargs = {"waypoints": [agent_pb2.Position(x=x, y=y) for x, y in waypoints]}
        if final_heading is not None:
            kwargs["final_heading"] = final_heading
        self._stub.SetPath(agent_pb2.Path(**kwargs))

    def set_escape_plan(self, waypoints: list[tuple[float, float]]) -> None:
        path = agent_pb2.Path(waypoints=[agent_pb2.Position(x=x, y=y) for x, y in waypoints])
        self._stub.SetEscapePlan(path)

    def observe_position(self, x: float, y: float, heading: float, confidence: float) -> None:
        self._stub.ObservePosition(agent_pb2.PositionObservation(x=x, y=y, heading=heading, confidence=confidence))

    def run_command(self, command: str, duration_s: float) -> None:
        self._stub.RunCommand(agent_pb2.CommandRequest(command=command, duration_s=duration_s))

    def scout(self) -> None:
        self._stub.Scout(agent_pb2.Empty())

    def stream_positions(self):
        for pos in self._stub.StreamPosition(agent_pb2.Empty()):
            yield pos.x, pos.y, pos.heading

    def stream_agent_updates(self):
        for update in self._stub.StreamAgentUpdate(agent_pb2.Empty()):
            camera_frame_jpg = update.camera_frame_jpg or None
            sensor_case = update.WhichOneof("sensor_readings")
            if sensor_case == "ray_batch":
                rays = [(r.start_x, r.start_y, r.end_x, r.end_y, r.did_collide)
                        for r in update.ray_batch.rays]
                yield camera_frame_jpg, rays, None
            elif sensor_case == "cone":
                yield camera_frame_jpg, None, (update.cone.ultrasonic_min_m, update.cone.heading)
            else:
                yield camera_frame_jpg, None, None

    def close(self) -> None:
        self._channel.close()
