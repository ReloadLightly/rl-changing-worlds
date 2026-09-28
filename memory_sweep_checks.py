"""Brief checks for newly configurable scales, starts, seeds, and the outcome."""

import json
from pathlib import Path

import numpy as np

from bandits import Environment, TRACKING_METHODS, STATIONARY_METHODS, make_seeds, simulate
from run_memory_sweep import METHODS, task_outcomes


def main():
    old = json.loads(Path("results/manifest.json").read_text())
    for i, kind in enumerate(("stationary", "random_walk", "sudden_change")):
        methods = STATIONARY_METHODS if i == 0 else TRACKING_METHODS
        assert make_seeds(old["master_seed"], i, methods) == old["experiments"][kind]["seeds"]
    seeds = make_seeds(20260928, 3, METHODS)
    assert seeds == make_seeds(20260928, 3, METHODS[::-1])
    assert len(set(seeds["agents"].values()) | set(seeds["environment"].values())) == 11

    shape = (8, 10)
    env_seeds = seeds["environment"]
    default = Environment("random_walk", *shape, env_seeds)
    assert (default.q == 0).all()
    normal = Environment("stationary", *shape, env_seeds)
    initial = np.random.default_rng(env_seeds["initial"]).normal(size=shape)
    np.testing.assert_array_equal(normal.q, initial)
    noise_rng = np.random.default_rng(env_seeds["rewards"])
    np.testing.assert_array_equal(normal.potential_rewards(), initial + noise_rng.normal(size=shape))

    envs = [Environment("random_walk", *shape, env_seeds, drift_std=d,
                        reward_std=r, initialization="normal")
            for d, r in ((0, 1), (0.003, 0.5), (0.01, 2))]
    reward_rng = np.random.default_rng(env_seeds["rewards"])
    drift_rng = np.random.default_rng(env_seeds["drift"])
    expected = [initial.copy() for _ in envs]
    for _ in range(20):
        z_reward = reward_rng.standard_normal(shape)
        z_drift = drift_rng.standard_normal(shape)
        for j, env in enumerate(envs):
            np.testing.assert_array_equal(env.q, expected[j])
            np.testing.assert_array_equal(env.potential_rewards(), env.q + env.reward_std * z_reward)
            env.after_interaction()
            expected[j] += env.drift_std * z_drift

    config = dict(kind="random_walk", tasks=8, arms=10, steps=20, bin_width=5,
                  drift_std=0.003, reward_std=2.0, initialization="normal", change_step=10)
    first = simulate(config, METHODS, seeds)
    reversed_methods = simulate(config, METHODS[::-1], seeds)
    np.testing.assert_array_equal(first["task_windows"], reversed_methods["task_windows"][::-1])
    np.testing.assert_array_equal(first["trace_q_true"][0], initial[0])
    outcome = task_outcomes(first)
    for m in range(len(METHODS)):
        true = first["trace_q_true"]
        actions = first["trace_actions"][m]
        regret = true.max(axis=1) - true[np.arange(config["steps"]), actions]
        np.testing.assert_allclose(outcome[m, 0, 0], regret.mean())
        residual = first["trace_rewards"][m] - true[np.arange(config["steps"]), actions]
        rng = np.random.default_rng(env_seeds["rewards"])
        expected_reward_noise = np.array([rng.normal(size=shape)[0, a] * 2 for a in actions])
        np.testing.assert_allclose(residual, expected_reward_noise, atol=1e-14)
    print("Passed: preserved seeds/defaults, new seeds, shared scaled draws including zero drift, "
          "configuration passthrough, method order, and per-decision regret.")


if __name__ == "__main__":
    main()
