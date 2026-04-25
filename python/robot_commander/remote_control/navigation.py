import math
import threading
import traceback
from typing import Callable

from robot_commander import WorldPosition2d
from robot_commander.localization.world_localizer import WorldPose
from robot_commander.map.map_coordinates import MapCoordinates
from robot_commander.remote_control.agent_client import AgentClient
from robot_commander.remote_control.obstacle_mapping import ObstacleMapper

_GOAL_REACHED_THRESHOLD_M = 0.1


class Navigator:
    def __init__(
        self,
        client: AgentClient,
        obstacle_mapper: ObstacleMapper,
        map_coords: MapCoordinates,
        get_agent_pos: Callable[[], WorldPose | None],
    ):
        self._client = client
        self._obstacle_mapper = obstacle_mapper
        self._map_coords = map_coords
        self._get_agent_pos = get_agent_pos

        self._planned_path: list[tuple[float, float]] = []
        self._checkpoint: tuple[float, float] | None = None
        self._goal_heading: float | None = None
        self._stop_seek_event = threading.Event()
        self._goal_seek_thread: threading.Thread | None = None

    def snapshot(self) -> tuple[list[tuple[float, float]], tuple[float, float] | None, float | None]:
        return list(self._planned_path), self._checkpoint, self._goal_heading

    def current_target(self) -> tuple[float, float] | None:
        if self._planned_path:
            return self._planned_path[-1]
        return self._checkpoint

    def handle_click(self, pixel_x: int, pixel_y: int, shift_held: bool, goal_heading: float | None = None) -> None:
        wx, wy = self._map_coords.px_to_world(pixel_x, pixel_y)
        if shift_held:
            agent_pos = self._get_agent_pos()
            if agent_pos is None:
                return
            self._goal_heading = goal_heading
            self._stop_seek_event.set()
            self._stop_seek_event = threading.Event()
            self._goal_seek_thread = threading.Thread(
                target=self._goal_seek,
                args=(wx, wy, goal_heading, self._stop_seek_event),
                daemon=True,
            )
            self._goal_seek_thread.start()
        else:
            self._stop_seek_event.set()
            self._goal_heading = None
            self._checkpoint = (wx, wy)
            self._planned_path = []
            self._client.set_checkpoint(wx, wy)

    def set_offset_waypoint(self, angle_offset_rad: float, distance_m: float) -> None:
        agent_pos = self._get_agent_pos()
        if agent_pos is None:
            return
        target_x = agent_pos.x + distance_m * math.cos(agent_pos.heading + angle_offset_rad)
        target_y = agent_pos.y + distance_m * math.sin(agent_pos.heading + angle_offset_rad)
        self._stop_seek_event.set()
        self._goal_heading = None
        self._checkpoint = (target_x, target_y)
        self._planned_path = []
        self._client.set_checkpoint(target_x, target_y)

    def _goal_seek(
        self,
        goal_x: float,
        goal_y: float,
        goal_heading: float | None,
        stop_event: threading.Event,
    ) -> None:
        if stop_event.is_set():
            return
        try:
            agent_pos = self._get_agent_pos()
            if agent_pos is None:
                return
            path = self._obstacle_mapper.plan_path(
                WorldPosition2d(agent_pos.x, agent_pos.y),
                WorldPosition2d(goal_x, goal_y),
                "user_path.npz",
            )
            if path is None:
                return
            self._planned_path = path
            self._checkpoint = None
            end_x, end_y = path[-1]
            dist_to_goal = math.sqrt((end_x - goal_x) ** 2 + (end_y - goal_y) ** 2)
            at_goal = dist_to_goal < _GOAL_REACHED_THRESHOLD_M
            tracking_path = path[1:] if len(path) > 1 else path
            self._client.set_path(tracking_path, final_heading=goal_heading if at_goal else None)
        except Exception:
            traceback.print_exc()
