# Simulation-only Robot Reliability OS MVP

A simulation-first MVP for testing a runtime reliability supervisor for robot manipulation without requiring physical hardware.

## What this MVP demonstrates

The system compares a baseline robot controller against a reliability-supervised controller under controlled disturbances.

The MVP currently includes:

- a lightweight 2D pick-and-place simulation
- deterministic and random failure/disturbance injection
- runtime risk estimation
- supervisor interventions and recovery actions
- quantitative experiment metrics
- JSON episode logging
- unit tests

The first goal is not photo-realistic robotics. It is to prove the **evaluation loop** and reliability concepts in a fast, reproducible environment. A later phase can swap the simulator adapter for PyBullet, MuJoCo, RLBench, or Isaac Sim while preserving the reliability and benchmarking layers.

## Architecture

```text
Simulation Environment
        |
        v
Baseline Controller
        |
        v
Reliability Supervisor (optional)
  - risk estimation
  - intervention policy
  - recovery action
        |
        v
Environment Step
        |
        v
Episode Metrics + Logs
```

## Disturbances

The MVP supports:

- object displacement
- sensor noise
- actuator noise
- reduced grip/friction
- obstacle insertion

## Metrics

- task success rate
- collision rate
- object drop rate
- intervention rate
- recovery success rate
- average completion steps
- mean predicted risk

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m robot_reliability.cli --episodes 200 --seed 7
```

Run tests:

```bash
pytest -q
```

## Example experiment

The CLI runs two matched experiments using the same disturbance seeds:

1. baseline controller only
2. baseline controller + reliability supervisor

This produces an apples-to-apples comparison of success, collisions, drops, recoveries, and intervention behavior.

## Project structure

```text
robot_reliability/
  config.py
  environment.py
  controller.py
  supervisor.py
  experiment.py
  metrics.py
  cli.py

tests/
  test_environment.py
  test_supervisor.py
  test_experiment.py
```

## Roadmap

1. MVP evaluation loop in lightweight simulation
2. Add learned failure prediction model
3. Add unseen-disturbance evaluation split
4. Add PyBullet/MuJoCo backend
5. Add multiple robot morphologies
6. Integrate POLARIS-style cross-embodiment skill transfer
7. Add dashboard and robot black-box incident analysis

## Important limitation

Results from this repository are **simulation results**. They validate software behavior under controlled assumptions; they do not prove real-world robot performance until hardware validation is performed.
