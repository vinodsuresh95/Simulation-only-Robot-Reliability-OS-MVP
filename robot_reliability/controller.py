from __future__ import annotations

import math
from typing import Tuple

from .config import SimulationConfig
from .environment import RobotState

Point = Tuple[float, float]


class BaselineController:
    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()

    def action(self, state: RobotState) -> Point:
        goal = state.target_position if state.holding_object else state.object_position
        return self._move_toward(state.position, goal, self.config.step_size)

    @staticmethod
    def _move_toward(start: Point, goal: Point, step_size: float) -> Point:
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        norm = math.hypot(dx, dy)
        if norm <= step_size or norm == 0:
            return goal
        return (start[0] + step_size * dx / norm, start[1] + step_size * dy / norm)
