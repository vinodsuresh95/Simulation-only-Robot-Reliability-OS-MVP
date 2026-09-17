from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .config import SimulationConfig
from .controller import BaselineController
from .environment import Disturbance, PickPlaceEnvironment
from .metrics import EpisodeResult, ExperimentSummary, summarize
from .supervisor import ReliabilitySupervisor


def sample_disturbance(rng: random.Random) -> tuple[str, Disturbance]:
    kind = rng.choice(["nominal", "object_shift", "sensor_noise", "actuator_noise", "low_grip", "obstacle", "compound"])
    if kind == "object_shift":
        return kind, Disturbance(object_displacement=(rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12)))
    if kind == "sensor_noise":
        return kind, Disturbance(sensor_noise=rng.uniform(0.08, 0.18))
    if kind == "actuator_noise":
        return kind, Disturbance(actuator_noise=rng.uniform(0.06, 0.14))
    if kind == "low_grip":
        return kind, Disturbance(grip_strength=rng.uniform(0.35, 0.60))
    if kind == "obstacle":
        return kind, Disturbance(obstacle=(rng.uniform(0.28, 0.72), rng.uniform(0.22, 0.65)))
    if kind == "compound":
        return kind, Disturbance(
            sensor_noise=rng.uniform(0.06, 0.13),
            actuator_noise=rng.uniform(0.05, 0.11),
            grip_strength=rng.uniform(0.40, 0.68),
            obstacle=(rng.uniform(0.30, 0.68), rng.uniform(0.24, 0.62)),
        )
    return kind, Disturbance()


def run_episode(episode: int, seed: int, disturbance: Disturbance, disturbance_name: str, supervised: bool) -> EpisodeResult:
    config = SimulationConfig()
    env = PickPlaceEnvironment(config=config, seed=seed)
    controller = BaselineController(config)
    supervisor = ReliabilitySupervisor(simulation=config)
    state = env.reset(disturbance)

    while state.step < config.max_steps and not state.success and not state.collision and not state.object_dropped:
        observed = env.observe()
        proposed = controller.action(observed)
        safe_mode = False
        if supervised:
            decision = supervisor.assess(observed, disturbance, proposed)
            state.risk_history.append(decision.risk)
            if decision.stop:
                state.interventions += 1
                break
            if decision.intervene:
                state.interventions += 1
                proposed = decision.action
                safe_mode = decision.safe_mode
        state = env.step(proposed, safe_mode=safe_mode)

    if supervised and state.interventions > 0 and state.success:
        state.recoveries += 1

    return EpisodeResult(
        episode=episode,
        success=state.success,
        collision=state.collision,
        object_dropped=state.object_dropped,
        steps=state.step,
        interventions=state.interventions,
        recoveries=state.recoveries,
        mean_risk=(sum(state.risk_history) / len(state.risk_history)) if state.risk_history else 0.0,
        disturbance=disturbance_name,
    )


def run_experiment(episodes: int = 100, seed: int = 7, supervised: bool = False) -> tuple[list[EpisodeResult], ExperimentSummary]:
    rng = random.Random(seed)
    results: list[EpisodeResult] = []
    for episode in range(episodes):
        name, disturbance = sample_disturbance(rng)
        episode_seed = rng.randint(0, 2**31 - 1)
        results.append(run_episode(episode, episode_seed, disturbance, name, supervised))
    return results, summarize(results)


def save_results(path: str | Path, results: list[EpisodeResult], summary: ExperimentSummary) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary.to_dict(), "episodes": [asdict(r) for r in results]}, indent=2))
