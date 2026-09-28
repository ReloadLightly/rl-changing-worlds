"""Brief checks of optimistic initialization, UCB priorities, and stable streams."""

import numpy as np

from bandits import Agent, Method, TRACKING_METHODS, make_seeds
from run_exploration import METHODS


def main():
    optimistic = Agent(16, 10, METHODS[3], 42)
    assert np.all(optimistic.q == 5) and np.all(optimistic.counts == 0)
    actions = optimistic.act()
    optimistic.learn(actions, np.zeros(16))
    assert np.all(optimistic.q[optimistic.rows, actions] == 4.5)
    assert np.all((optimistic.q == 5).sum(axis=1) == 9)

    # Even huge observed rewards cannot defeat the priority for untried actions.
    for method in METHODS[4:]:
        agent = Agent(2000, 10, method, 42)
        with np.errstate(divide="raise", invalid="raise"):
            for t in range(10):
                actions = agent.act()
                assert np.all(agent.counts[agent.rows, actions] == 0)
                if t == 0:
                    assert np.all(np.abs(np.bincount(actions, minlength=10) - 200) < 60)
                agent.learn(actions, np.full(2000, 1e6))
        assert np.all(agent.counts == 1) and agent.time == 10

    # Exact finite-count formula and random ties once all actions were tried.
    agent = Agent(2000, 2, METHODS[4], 7)
    agent.time = 10
    agent.counts[:] = [1, 9]
    bonus = 2 * np.sqrt(np.log(11) / np.array([1, 9]))
    agent.q[:] = -bonus  # Equal UCB indices, despite unequal estimates/counts.
    assert 850 < np.count_nonzero(agent.act() == 0) < 1150
    agent.q[:, 1] += 1e-6
    assert np.all(agent.act() == 1)

    # Explicit zero initialization remains the default; added methods cannot
    # perturb streams for existing methods, even when their order changes.
    assert np.all(Agent(2, 10, Method("zero", 0.1), 1).q == 0)
    old = make_seeds(20260928, 1, TRACKING_METHODS)
    extended = make_seeds(20260928, 1, list(reversed(METHODS + TRACKING_METHODS)))
    assert old["environment"] == extended["environment"]
    assert all(seed == extended["agents"][name] for name, seed in old["agents"].items())
    print("Optimistic initialization, explicit UCB priorities, formula, ties, and seeds: OK")


if __name__ == "__main__":
    main()
