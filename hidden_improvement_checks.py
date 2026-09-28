"""Brief intervention, pairing, and censored-exposure checks."""

import numpy as np

from bandits import Environment, TargetVisits, make_seeds, simulate
from run_hidden_improvement import METHODS, median_delay, unvisited_curve


def main():
    seeds = make_seeds(20260928, 7, METHODS)
    a = Environment("unchanged", 16, 10, seeds["environment"], change_step=2)
    b = Environment("hidden_improvement", 16, 10, seeds["environment"], change_step=2)
    initial = a.q.copy()
    np.testing.assert_array_equal(a.target, initial.argmin(axis=1))
    for t in range(4):
        a.before_action(t)
        b.before_action(t)
        expected = initial.copy()
        if t >= 2:
            expected[np.arange(16), a.target] = initial.max(axis=1) + 0.5
        np.testing.assert_array_equal(b.q, expected)
        np.testing.assert_allclose(a.potential_rewards() - a.q, b.potential_rewards() - b.q)

    visits = TargetVisits(np.array([0, 1, 2]), 1, 2, 5)
    visits.record(0, 1, np.array([0, 1, 0]))
    visits.record(0, 2, np.array([0, 0, 0]))  # Immediate exposure, delay 0.
    visits.record(0, 4, np.array([0, 1, 0]))  # Delay 2, last decision; task 2 censored.
    np.testing.assert_array_equal(visits.pre[0], [1, 1, 0])
    np.testing.assert_array_equal(visits.first_delay[0], [0, 2, -1])
    np.testing.assert_array_equal(visits.first_window[0], [2, 1, 0])
    survival, _, _ = unvisited_curve(visits.first_delay[0], 3)
    np.testing.assert_allclose(survival, [1, 2/3, 2/3, 1/3])
    assert median_delay(np.array([0, -1, -1]), 3) is None
    assert median_delay(np.array([0, 2, -1]), 3) == 2

    config = dict(tasks=16, arms=10, steps=30, bin_width=5, drift_std=0,
                  reward_std=1, change_step=15, initialization="normal")
    runs = [simulate(dict(config, kind=kind), METHODS, seeds)
            for kind in ("unchanged", "hidden_improvement")]
    for key in ("target_arms", "target_pre_visits", "target_first_delay"):
        np.testing.assert_array_equal(runs[0][key], runs[1][key])
    np.testing.assert_array_equal(runs[0]["trace_actions"][:, :15], runs[1]["trace_actions"][:, :15])
    print("Intervention, matched worlds/noise, immediate visits, censoring, and pairing: OK")


if __name__ == "__main__":
    main()
