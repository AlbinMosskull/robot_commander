import math

import cv2
import numpy as np

from robot_commander.map.map_coordinates import MapCoordinates
from robot_commander.remote_control.controller import MapState

_FREE_THRESHOLD = 0.3
_OCCUPIED_THRESHOLD = 0.7
_OVERLAY_ALPHA = 0.45


class MapRenderer:
    def __init__(self, map_coords: MapCoordinates):
        self._map_coords = map_coords

    def render(self, state: MapState) -> np.ndarray:
        canvas = self._map_coords.background.copy()
        self._draw_occupancy_overlay(canvas, state.occ_grid)

        if state.planned_path:
            pts = [self._map_coords.world_to_px(wx, wy) for wx, wy in state.planned_path]
            for point_a, point_b in zip(pts, pts[1:]):
                cv2.line(canvas, point_a, point_b, (0, 200, 0), 2)
            cv2.circle(canvas, pts[-1], 8, (0, 200, 0), -1)
            cv2.circle(canvas, pts[-1], 8, (0, 0, 0), 1)
        elif state.checkpoint is not None:
            checkpoint_px = self._map_coords.world_to_px(*state.checkpoint)
            cv2.circle(canvas, checkpoint_px, 8, (0, 200, 0), -1)
            cv2.circle(canvas, checkpoint_px, 8, (0, 0, 0), 1)

        if state.agent_pos is not None:
            agent_px = self._map_coords.world_to_px(state.agent_pos.x, state.agent_pos.y)
            cv2.circle(canvas, agent_px, 8, (200, 80, 0), -1)
            cv2.circle(canvas, agent_px, 8, (0, 0, 0), 1)
            arrow_len_m = 0.15
            tip_px = self._map_coords.world_to_px(
                state.agent_pos.x + arrow_len_m * math.cos(state.agent_pos.heading),
                state.agent_pos.y + arrow_len_m * math.sin(state.agent_pos.heading),
            )
            cv2.arrowedLine(canvas, agent_px, tip_px, (200, 80, 0), 2, tipLength=0.4)

        return canvas

    def _draw_occupancy_overlay(self, canvas: np.ndarray, occ_grid: np.ndarray) -> None:
        grid = np.flipud(occ_grid)
        grid = cv2.resize(grid, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = canvas.copy()
        overlay[grid < _FREE_THRESHOLD] = (0, 200, 0)
        overlay[grid > _OCCUPIED_THRESHOLD] = (0, 0, 200)

        known = (grid < _FREE_THRESHOLD) | (grid > _OCCUPIED_THRESHOLD)
        blended = cv2.addWeighted(overlay, _OVERLAY_ALPHA, canvas, 1 - _OVERLAY_ALPHA, 0)
        canvas[known] = blended[known]
