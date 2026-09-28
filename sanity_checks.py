"""A few scientific sanity checks. Run with: python sanity_checks.py."""

import numpy as np

from bandits import Agent, Environment, Method, TRACKING_METHODS, action_metrics, make_seeds, simulate


def main():
    for alpha, expected in ((None, 3.0), (0.1, 0.58)):
        agent = Agent(1, 3, Method("check", 0, alpha), 1)
        agent.learn(np.array([1]), np.array([2.0]))
        agent.learn(np.array([1]), np.array([4.0]))
        np.testing.assert_allclose(agent.q, [[0, expected, 0]])
        np.testing.assert_array_equal(agent.counts, [[0, 2, 0]])

    agent = Agent(30000, 3, Method("ties", 0), 2)
    agent.q[:, 1] = -1
    counts = np.bincount(agent.act(), minlength=3)
    assert counts[1] == 0 and abs(counts[0] / 30000 - 0.5) < 0.02
    optimal, regret = action_metrics(np.array([[2, 2, 0], [0, 0, 0]]), np.array([1, 2]))
    assert optimal.all() and (regret == 0).all()

    seeds = make_seeds(123, 1, TRACKING_METHODS)
    env = Environment("random_walk", 10000, 10, seeds["environment"])
    assert (env.q == 0).all()
    rewards = env.potential_rewards()
    assert abs(rewards.mean()) < 0.02 and abs(rewards.std() - 1) < 0.02
    env.after_interaction()
    assert abs(env.q.mean()) < 0.0002 and abs(env.q.std() - 0.01) < 0.0002

    env = Environment("sudden_change", 32, 10, seeds["environment"], change_step=10)
    initial = env.q.copy()
    env.before_action(9)
    np.testing.assert_array_equal(env.q, initial)
    env.before_action(10)
    np.testing.assert_array_equal(env.q, np.take_along_axis(initial, env.permutation, axis=1))
    np.testing.assert_array_equal(np.sort(env.q, axis=1), np.sort(initial, axis=1))

    config = dict(kind="random_walk", tasks=8, arms=10, steps=20,
                  bin_width=5, drift_std=0.01, change_step=10)
    first = simulate(config, TRACKING_METHODS, seeds)
    reordered = simulate(config, TRACKING_METHODS[::-1], seeds)
    for key in ("curve_mean", "bin_sem", "task_windows", "trace_actions", "trace_estimates"):
        np.testing.assert_array_equal(first[key], reordered[key][::-1])
    np.testing.assert_array_equal(first["trace_q_true"], reordered["trace_q_true"])
    # Reconstruct logged regret directly from the selected actions and true values.
    for m in range(len(TRACKING_METHODS)):
        true = first["trace_q_true"]
        chosen = true[np.arange(config["steps"]), first["trace_actions"][m]]
        expected = (true.max(axis=1) - chosen).sum()
        np.testing.assert_allclose(first["task_windows"][m, 0, 0, 2], expected)
    print("Passed: updates, random ties, tied optima, reward/drift scales, switch timing, "
          "shared worlds, deterministic replay, and regret accounting.")


if __name__ == "__main__":
    main()
