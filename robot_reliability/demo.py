from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .config import SimulationConfig
from .controller import BaselineController
from .environment import Disturbance, PickPlaceEnvironment
from .experiment import run_experiment
from .supervisor import ReliabilitySupervisor
from .visualization import Frame, save_comparison_chart, save_episode_gif


def record_demo(seed: int = 11) -> tuple[list[Frame], Disturbance]:
    config = SimulationConfig()
    disturbance = Disturbance(sensor_noise=.08, actuator_noise=.07, grip_strength=.62, obstacle=(.58, .52))
    env = PickPlaceEnvironment(config, seed=seed)
    controller = BaselineController(config)
    supervisor = ReliabilitySupervisor(simulation=config)
    state = env.reset(disturbance)
    frames: list[Frame] = [Frame(copy.deepcopy(state), 0, False, False, "Task started")]

    while state.step < config.max_steps and not state.success and not state.collision and not state.object_dropped:
        observed = env.observe()
        proposed = controller.action(observed)
        decision = supervisor.assess(observed, disturbance, proposed)
        state.risk_history.append(decision.risk)
        event = "Nominal execution"
        safe_mode = False
        if decision.stop:
            state.interventions += 1
            frames.append(Frame(copy.deepcopy(state), decision.risk, True, True, "Unsafe action stopped"))
            break
        if decision.intervene:
            state.interventions += 1
            proposed = decision.action
            safe_mode = decision.safe_mode
            event = "Risk detected -> safer recovery action"
        state = env.step(proposed, safe_mode=safe_mode)
        if state.success and state.interventions:
            state.recoveries += 1
            event = "Task recovered successfully"
        frames.append(Frame(copy.deepcopy(state), decision.risk, decision.intervene, safe_mode, event))
    return frames, disturbance


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Robot Reliability OS visual demo assets")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    frames, disturbance = record_demo(args.seed)
    gif = save_episode_gif(frames, disturbance, out / "reliability_demo.gif")
    _, baseline = run_experiment(args.episodes, args.seed, supervised=False)
    _, supervised = run_experiment(args.episodes, args.seed, supervised=True)
    chart = save_comparison_chart(baseline.to_dict(), supervised.to_dict(), out / "baseline_vs_reliability.png")

    print(f"Demo: {gif}")
    print(f"Chart: {chart}")
    print("Baseline:", baseline.to_dict())
    print("Reliability OS:", supervised.to_dict())


if __name__ == "__main__":
    main()
