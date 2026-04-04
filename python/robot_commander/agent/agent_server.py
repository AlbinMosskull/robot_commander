import time
from concurrent import futures

import grpc

from robot_commander.agent.abstract_agent import AbstractAgent
from robot_commander.agent.simulated.simulated_agent import SimulatedAgent
from robot_commander.proto import agent_pb2, agent_pb2_grpc
from robot_commander import config as cfg

_STREAM_HZ = 10


class AgentControlServicer(agent_pb2_grpc.AgentControlServicer):
    def __init__(self, agent: AbstractAgent):
        self._agent = agent

    def SetCheckpoint(self, request, context):
        self._agent.SetWaypointList([(request.x, request.y)])
        return agent_pb2.Empty()

    def SetPath(self, request, context):
        self._agent.SetWaypointList([(p.x, p.y) for p in request.waypoints])
        return agent_pb2.Empty()

    def StreamPosition(self, request, context):
        while context.is_active():
            x, y = self._agent.GetXandY()
            yield agent_pb2.Position(x=x, y=y)
            time.sleep(1 / _STREAM_HZ)

    def StreamRays(self, request, context):
        while context.is_active():
            readings = self._agent.GetSensorReading()
            if readings:
                yield agent_pb2.RayBatch(rays=[
                    agent_pb2.Ray(
                        start_x=r.start_x, start_y=r.start_y,
                        end_x=r.end_x, end_y=r.end_y,
                        did_collide=r.did_hit,
                    )
                    for r in readings
                ])
            time.sleep(1 / _STREAM_HZ)


def main():
    port = cfg.load().agent.port
    agent = SimulatedAgent()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    agent_pb2_grpc.add_AgentControlServicer_to_server(AgentControlServicer(agent), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Agent server listening on port {port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=None)


if __name__ == "__main__":
    main()
