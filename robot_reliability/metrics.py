from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    success: bool
    collision: bool
    object_dropped: bool
    steps: int
    interventions: int
    recoveries: int
    mean_risk: float
    disturbance: str


@dataclass(frozen=True)
class ExperimentSummary:
    episodes: int
    success_rate: float
    collision_rate: float
    drop_rate: float
    intervention_rate: float
    recovery_success_rate: float
    average_steps: float
    mean_risk: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(results: list[EpisodeResult]) -> ExperimentSummary:
    if not results:
        return ExperimentSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    episodes = len(results)
    interventions = sum(r.interventions for r in results)
    recovered = sum(1 for r in results if r.recoveries > 0 and r.success)
    episodes_with_intervention = sum(1 for r in results if r.interventions > 0)
    return ExperimentSummary(
        episodes=episodes,
        success_rate=sum(r.success for r in results) / episodes,
        collision_rate=sum(r.collision for r in results) / episodes,
        drop_rate=sum(r.object_dropped for r in results) / episodes,
        intervention_rate=episodes_with_intervention / episodes,
        recovery_success_rate=(recovered / episodes_with_intervention) if episodes_with_intervention else 0.0,
        average_steps=mean(r.steps for r in results),
        mean_risk=mean(r.mean_risk for r in results),
    )
