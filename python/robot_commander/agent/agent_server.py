import time
from concurrent import futures

import cv2
import grpc

from robot_commander.agent.abstract_agent import AbstractAgent
from robot_commander.agent.simulated.simulated_agent import SimulatedAgent
from robot_commander.agent.adeept.adeept_agent import AdeeptAgent
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
        final_heading = request.final_heading if request.HasField("final_heading") else None
        self._agent.SetWaypointList([(p.x, p.y) for p in request.waypoints], final_heading=final_heading)
        return agent_pb2.Empty()

    def SetEscapePlan(self, request, context):
        self._agent.SetEscapePlan([(p.x, p.y) for p in request.waypoints])
        return agent_pb2.Empty()

    def ObservePosition(self, request, context):
        self._agent.ObservePosition(request.x, request.y, request.heading, request.confidence)
        return agent_pb2.Empty()

    def StreamPosition(self, request, context):
        while context.is_active():
            x, y = self._agent.GetXandY()
            yield agent_pb2.Position(x=x, y=y, heading=self._agent.GetHeading())
            time.sleep(1 / _STREAM_HZ)

    def RunCommand(self, request, context):
        self._agent.RunCommand(request.command, request.duration_s)
        return agent_pb2.Empty()

    def Scout(self, request, context):
        self._agent.Scout()
        return agent_pb2.Empty()

    def EnablePayload(self, request, context):
        self._agent.EnablePayload()
        return agent_pb2.Empty()

    def StreamAgentUpdate(self, request, context):
        while context.is_active():
            frame = self._agent.GetCameraReading()
            camera_frame_jpg = b""
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    camera_frame_jpg = buf.tobytes()

            pending_payload = self._agent.GetPayload()
            kwargs = {}
            if pending_payload is not None:
                kwargs["payload_frame_jpg"] = pending_payload

            ultrasonic_min = self._agent.GetUltrasonicMin()
            if ultrasonic_min is not None:
                yield agent_pb2.AgentUpdate(
                    camera_frame_jpg=camera_frame_jpg,
                    cone=agent_pb2.ConeReading(
                        ultrasonic_min_m=ultrasonic_min,
                        heading=self._agent.GetHeading(),
                    ),
                    **kwargs,
                )
            else:
                readings = self._agent.GetSensorReading()
                if readings:
                    yield agent_pb2.AgentUpdate(
                        camera_frame_jpg=camera_frame_jpg,
                        ray_batch=agent_pb2.RayBatch(rays=[
                            agent_pb2.Ray(
                                start_x=r.start_x, start_y=r.start_y,
                                end_x=r.end_x, end_y=r.end_y,
                                did_collide=r.did_hit,
                            )
                            for r in readings
                        ]),
                        **kwargs,
                    )
                else:
                    yield agent_pb2.AgentUpdate(camera_frame_jpg=camera_frame_jpg, **kwargs)

            time.sleep(1 / _STREAM_HZ)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-escape-plan", action="store_true", help="Disable escape plan")
    parser.add_argument("--raw-sensor", action="store_true", help="Sweep ultrasonic sensor and stream rays")
    args = parser.parse_args()

    port = cfg.load().agent.port
    agent = AdeeptAgent(escape_plan_enabled=not args.no_escape_plan, raw_sensor=args.raw_sensor)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    agent_pb2_grpc.add_AgentControlServicer_to_server(AgentControlServicer(agent), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Agent server listening on port {port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=None)
        agent.close()


if __name__ == "__main__":
    main()
