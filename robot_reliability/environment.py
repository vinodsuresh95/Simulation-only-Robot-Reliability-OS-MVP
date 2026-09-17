from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Optional, Tuple

from .config import SimulationConfig

Point = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class Disturbance:
    object_displacement: Point = (0.0, 0.0)
    sensor_noise: float = 0.0
    actuator_noise: float = 0.0
    grip_strength: float = 1.0
    obstacle: Optional[Point] = None


@dataclass
class RobotState:
    position: Point = (0.10, 0.10)
    object_position: Point = (0.55, 0.35)
    target_position: Point = (0.85, 0.82)
    holding_object: bool = False
    object_dropped: bool = False
    collision: bool = False
    success: bool = False
    step: int = 0
    interventions: int = 0
    recoveries: int = 0
    risk_history: list[float] = field(default_factory=list)


class PickPlaceEnvironment:
    """Small deterministic-friendly simulator used to validate reliability logic."""

    def __init__(self, config: SimulationConfig | None = None, seed: int = 0):
        self.config = config or SimulationConfig()
        self.rng = random.Random(seed)
        self.disturbance = Disturbance()
        self.state = RobotState()

    def reset(self, disturbance: Disturbance | None = None) -> RobotState:
        self.disturbance = disturbance or Disturbance()
        displaced = (
            0.55 + self.disturbance.object_displacement[0],
            0.35 + self.disturbance.object_displacement[1],
        )
        self.state = RobotState(object_position=self._clip(displaced))
        return self.state

    def observe(self) -> RobotState:
        noise = self.disturbance.sensor_noise
        if noise <= 0:
            return self.state
        observed_object = (
            self.state.object_position[0] + self.rng.gauss(0, noise),
            self.state.object_position[1] + self.rng.gauss(0, noise),
        )
        return RobotState(
            position=self.state.position,
            object_position=self._clip(observed_object),
            target_position=self.state.target_position,
            holding_object=self.state.holding_object,
            object_dropped=self.state.object_dropped,
            collision=self.state.collision,
            success=self.state.success,
            step=self.state.step,
            interventions=self.state.interventions,
            recoveries=self.state.recoveries,
            risk_history=list(self.state.risk_history),
        )

    def step(self, desired_position: Point, safe_mode: bool = False) -> RobotState:
        self.state.step += 1
        desired_position = self._clip(desired_position)
        noise_scale = self.disturbance.actuator_noise * (0.25 if safe_mode else 1.0)
        actual = (
            desired_position[0] + self.rng.gauss(0, noise_scale),
            desired_position[1] + self.rng.gauss(0, noise_scale),
        )
        self.state.position = self._clip(actual)

        obstacle = self.disturbance.obstacle
        if obstacle and distance(self.state.position, obstacle) < self.config.collision_radius:
            self.state.collision = True

        if not self.state.holding_object and not self.state.object_dropped:
            if distance(self.state.position, self.state.object_position) <= self.config.grasp_radius:
                grasp_probability = max(0.0, min(1.0, self.disturbance.grip_strength))
                if safe_mode:
                    grasp_probability = min(1.0, grasp_probability + 0.25)
                if self.rng.random() <= grasp_probability:
                    self.state.holding_object = True

        if self.state.holding_object:
            self.state.object_position = self.state.position
            drop_probability = max(0.0, (1.0 - self.disturbance.grip_strength) * (0.25 if safe_mode else 0.65))
            if self.rng.random() < drop_probability:
                self.state.holding_object = False
                self.state.object_dropped = True

        if self.state.holding_object and distance(self.state.position, self.state.target_position) <= self.config.place_radius:
            self.state.success = True

        return self.state

    def _clip(self, point: Point) -> Point:
        lo, hi = self.config.workspace_min, self.config.workspace_max
        return (max(lo, min(hi, point[0])), max(lo, min(hi, point[1])))
