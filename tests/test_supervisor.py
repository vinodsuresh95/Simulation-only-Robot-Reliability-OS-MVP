from robot_reliability.environment import Disturbance, RobotState
from robot_reliability.supervisor import ReliabilitySupervisor


def test_nominal_state_does_not_intervene():
    supervisor = ReliabilitySupervisor()
    state = RobotState()
    decision = supervisor.assess(state, Disturbance(), (0.2, 0.2))
    assert decision.risk < 0.55
    assert not decision.intervene


def test_compound_risk_triggers_intervention():
    supervisor = ReliabilitySupervisor()
    state = RobotState(position=(0.1, 0.1))
    disturbance = Disturbance(sensor_noise=0.14, actuator_noise=0.12, grip_strength=0.35, obstacle=(0.16, 0.16))
    decision = supervisor.assess(state, disturbance, (0.2, 0.2))
    assert decision.risk >= 0.55
    assert decision.intervene
