from robot_reliability.environment import Disturbance, PickPlaceEnvironment


def test_object_displacement_is_applied():
    env = PickPlaceEnvironment(seed=1)
    state = env.reset(Disturbance(object_displacement=(0.1, -0.1)))
    assert state.object_position == (0.65, 0.24999999999999997)


def test_nominal_environment_can_grasp():
    env = PickPlaceEnvironment(seed=1)
    state = env.reset()
    env.step(state.object_position)
    assert env.state.holding_object
