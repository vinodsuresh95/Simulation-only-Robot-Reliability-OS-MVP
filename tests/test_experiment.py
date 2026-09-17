from robot_reliability.experiment import run_experiment


def test_experiment_is_reproducible():
    first, summary_a = run_experiment(episodes=20, seed=11, supervised=False)
    second, summary_b = run_experiment(episodes=20, seed=11, supervised=False)
    assert first == second
    assert summary_a == summary_b


def test_supervised_experiment_records_risk():
    results, summary = run_experiment(episodes=20, seed=13, supervised=True)
    assert len(results) == 20
    assert summary.episodes == 20
    assert any(result.mean_risk > 0 for result in results)
