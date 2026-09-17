from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from .config import ReliabilityConfig, SimulationConfig
from .environment import Disturbance, RobotState, distance

Point = Tuple[float, float]


@dataclass(frozen=True)
class SupervisorDecision:
    risk: float
    intervene: bool
    stop: bool
    reason: str
    action: Point
    safe_mode: bool


class ReliabilitySupervisor:
    """Rule-based v1 supervisor. Later versions can replace risk scoring with ML."""

    def __init__(self, reliability: ReliabilityConfig | None = None, simulation: SimulationConfig | None = None):
        self.config = reliability or ReliabilityConfig()
        self.simulation = simulation or SimulationConfig()

    def assess(self, state: RobotState, disturbance: Disturbance, proposed_action: Point) -> SupervisorDecision:
        components: list[tuple[str, float]] = []

        sensor_risk = min(1.0, disturbance.sensor_noise / max(self.config.sensor_noise_warning, 1e-6))
        actuator_risk = min(1.0, disturbance.actuator_noise / max(self.config.actuator_noise_warning, 1e-6))
        grip_risk = max(0.0, min(1.0, (self.config.grip_warning - disturbance.grip_strength) / self.config.grip_warning))
        components.extend([("sensor", sensor_risk), ("actuator", actuator_risk), ("grip", grip_risk)])

        obstacle_risk = 0.0
        if disturbance.obstacle:
            d = self._point_segment_distance(disturbance.obstacle, state.position, proposed_action)
            obstacle_risk = max(0.0, min(1.0, 1.0 - d / self.config.obstacle_warning_distance))
            components.append(("obstacle", obstacle_risk))

        risk = min(1.0, 0.28 * sensor_risk + 0.28 * actuator_risk + 0.24 * grip_risk + 0.45 * obstacle_risk)
        dominant = max(components, key=lambda item: item[1])
        intervene = risk >= self.config.intervene_threshold
        stop = risk >= self.config.stop_threshold
        action = proposed_action
        reason = dominant[0] if intervene else "nominal"

        if intervene and disturbance.obstacle and obstacle_risk >= 0.45:
            action = self._detour(state.position, proposed_action, disturbance.obstacle)
        elif intervene:
            # Conservative progress reduces actuator-error and improves grasp probability in safe mode.
            action = (
                state.position[0] + 0.5 * (proposed_action[0] - state.position[0]),
                state.position[1] + 0.5 * (proposed_action[1] - state.position[1]),
            )

        return SupervisorDecision(risk, intervene, stop, reason, action, intervene)

    def _detour(self, start: Point, goal: Point, obstacle: Point) -> Point:
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        norm = math.hypot(dx, dy) or 1.0
        perpendicular = (-dy / norm, dx / norm)
        offset = self.config.obstacle_warning_distance
        candidates = [
            (obstacle[0] + perpendicular[0] * offset, obstacle[1] + perpendicular[1] * offset),
            (obstacle[0] - perpendicular[0] * offset, obstacle[1] - perpendicular[1] * offset),
        ]
        candidate = min(candidates, key=lambda p: distance(start, p) + distance(p, goal))
        return (
            max(self.simulation.workspace_min, min(self.simulation.workspace_max, candidate[0])),
            max(self.simulation.workspace_min, min(self.simulation.workspace_max, candidate[1])),
        )

    @staticmethod
    def _point_segment_distance(point: Point, a: Point, b: Point) -> float:
        ab = (b[0] - a[0], b[1] - a[1])
        ap = (point[0] - a[0], point[1] - a[1])
        denom = ab[0] ** 2 + ab[1] ** 2
        if denom == 0:
            return distance(point, a)
        t = max(0.0, min(1.0, (ap[0] * ab[0] + ap[1] * ab[1]) / denom))
        projection = (a[0] + t * ab[0], a[1] + t * ab[1])
        return distance(point, projection)
