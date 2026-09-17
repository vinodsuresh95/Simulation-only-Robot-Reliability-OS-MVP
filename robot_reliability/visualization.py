from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from .environment import Disturbance, RobotState


@dataclass
class Frame:
    state: RobotState
    risk: float
    intervention: bool
    safe_mode: bool
    event: str


def _status(frame: Frame) -> str:
    if frame.state.success:
        return "RECOVERY SUCCESS" if frame.state.interventions else "TASK SUCCESS"
    if frame.state.collision:
        return "COLLISION"
    if frame.state.object_dropped:
        return "OBJECT DROPPED"
    if frame.intervention:
        return "SUPERVISOR INTERVENING"
    return "EXECUTING"


def save_episode_gif(frames: Iterable[Frame], disturbance: Disturbance, output: str | Path, title: str = "Robot Reliability OS") -> Path:
    frames = list(frames)
    if not frames:
        raise ValueError("At least one frame is required")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, 0])
    panel = fig.add_subplot(gs[0, 1])

    def draw(i: int) -> None:
        ax.clear(); panel.clear()
        f = frames[i]; s = f.state
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(title); ax.set_xlabel("Simulated workspace")
        ax.grid(alpha=.2)
        ax.add_patch(Rectangle((s.target_position[0]-.06, s.target_position[1]-.06), .12, .12, fill=False, linewidth=2))
        ax.add_patch(Circle(s.object_position, .025, alpha=.8))
        ax.add_patch(Circle(s.position, .035, alpha=.9))
        ax.plot([.1, s.position[0]], [.1, s.position[1]], linewidth=5, alpha=.55)
        if disturbance.obstacle:
            ax.add_patch(Circle(disturbance.obstacle, .055, alpha=.55))
            ax.text(disturbance.obstacle[0], disturbance.obstacle[1]+.075, "Obstacle", ha="center", fontsize=8)
        ax.text(s.position[0], s.position[1]+.06, "Robot EE", ha="center", fontsize=8)
        ax.text(s.target_position[0], s.target_position[1]+.08, "Target", ha="center", fontsize=8)

        panel.axis("off")
        risk_label = "LOW" if f.risk < .35 else "MEDIUM" if f.risk < .65 else "HIGH"
        lines = [
            "RUNTIME SUPERVISOR",
            "",
            f"Status: {_status(f)}",
            f"Step: {s.step}",
            f"Risk: {f.risk:.0%} ({risk_label})",
            f"Interventions: {s.interventions}",
            f"Recovery mode: {'ON' if f.safe_mode else 'OFF'}",
            f"Holding object: {'YES' if s.holding_object else 'NO'}",
            "",
            f"Event: {f.event}",
            "",
            "Disturbance telemetry",
            f"Sensor noise: {disturbance.sensor_noise:.2f}",
            f"Actuator noise: {disturbance.actuator_noise:.2f}",
            f"Grip strength: {disturbance.grip_strength:.2f}",
            f"Object shift: {disturbance.object_displacement}",
        ]
        panel.text(.03, .97, "\n".join(lines), va="top", family="monospace", fontsize=11)
        panel.add_patch(Rectangle((.03, .08), .9, .045, fill=False, transform=panel.transAxes))
        panel.add_patch(Rectangle((.03, .08), .9 * min(1, f.risk), .045, alpha=.6, transform=panel.transAxes))
        panel.text(.03, .14, "Failure risk", transform=panel.transAxes, fontsize=10)

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=350, repeat_delay=1200)
    animation.save(output, writer=PillowWriter(fps=3))
    plt.close(fig)
    return output


def save_comparison_chart(baseline: dict, supervised: dict, output: str | Path) -> Path:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    names = ["Success", "Collision", "Drops"]
    b = [baseline["success_rate"], baseline["collision_rate"], baseline["drop_rate"]]
    s = [supervised["success_rate"], supervised["collision_rate"], supervised["drop_rate"]]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i-.18 for i in x], b, width=.36, label="Baseline")
    ax.bar([i+.18 for i in x], s, width=.36, label="Reliability OS")
    ax.set_xticks(list(x), names); ax.set_ylim(0, 1); ax.set_ylabel("Rate")
    ax.set_title("Baseline vs Robot Reliability OS")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)
    return output
