from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    workspace_min: float = 0.0
    workspace_max: float = 1.0
    step_size: float = 0.08
    grasp_radius: float = 0.09
    place_radius: float = 0.10
    max_steps: int = 40
    collision_radius: float = 0.10


@dataclass(frozen=True)
class ReliabilityConfig:
    intervene_threshold: float = 0.55
    stop_threshold: float = 0.90
    sensor_noise_warning: float = 0.10
    actuator_noise_warning: float = 0.08
    grip_warning: float = 0.55
    obstacle_warning_distance: float = 0.18
