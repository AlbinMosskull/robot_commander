import math
import time
from concurrent import futures

import grpc
import pytest

from robot_commander.agent.agent_server import AgentControlServicer
from robot_commander.agent.simulated.sensors import SimulatedSensor
from robot_commander.agent.simulated.simulated_agent import SimulatedAgent
from robot_commander.proto import agent_pb2_grpc
from robot_commander.remote_control.agent_client import AgentClient
from robot_commander.sensor.range_reading import RangeReading


class NullSensor(SimulatedSensor):
    def read(self, x: float, y: float, heading: float) -> list[RangeReading]:
        return [RangeReading(x, y, x + 1.0, y, False)]


@pytest.fixture(scope="module")
def agent_server_and_client():
    agent = SimulatedAgent(sensor=NullSensor())
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    agent_pb2_grpc.add_AgentControlServicer_to_server(AgentControlServicer(agent), server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    client = AgentClient(host="localhost", port=port)
    yield agent, client
    server.stop(grace=0)
    client.close()


def test_position_stream_delivers_continuous_updates(agent_server_and_client):
    _, client = agent_server_and_client
    collected_positions = []
    for x, y, heading in client.stream_positions():
        collected_positions.append((x, y, heading))
        if len(collected_positions) >= 5:
            break
    assert len(collected_positions) == 5
    for x, y, heading in collected_positions:
        assert math.isfinite(x)
        assert math.isfinite(y)
        assert math.isfinite(heading)


def test_agent_moves_toward_waypoint(agent_server_and_client):
    agent, client = agent_server_and_client
    initial_x, initial_y = agent.GetXandY()
    client.set_path([(0.0, initial_y + 1.0)])
    time.sleep(1.5)
    final_x, final_y = agent.GetXandY()
    assert final_y > initial_y + 0.02


def test_agent_update_stream_delivers_ray_batch(agent_server_and_client):
    _, client = agent_server_and_client
    collected_updates = []
    for camera_frame, rays, cone, payload_frame in client.stream_agent_updates():
        collected_updates.append((camera_frame, rays, cone, payload_frame))
        if len(collected_updates) >= 3:
            break
    assert len(collected_updates) == 3
