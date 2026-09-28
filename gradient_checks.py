"""Brief checks of Section 2.8's policy, preference update, and baseline timing."""

import numpy as np

from gradient_bandits import GradientAgent, METHODS, softmax


def main():
    h = np.array([[1000., 1001., 1002.], [-1000., -1001., -1002.]])
    with np.errstate(over="raise", invalid="raise"):
        p = softmax(h)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_allclose(p, softmax(h + 10000))
    np.testing.assert_array_equal(softmax(np.array([[10000., 0., -10000.]])), [[1., 0., 0.]])

    agent = GradientAgent(40000, 3, METHODS[0], 42)
    agent.preferences[:] = np.log([.1, .3, .6])
    actions = agent.act()
    np.testing.assert_allclose(np.bincount(actions, minlength=3) / len(actions), [.1, .3, .6], atol=.008)

    # Separate task baselines; the first reward produces no preference change.
    agent = GradientAgent(2, 3, METHODS[0], 7)
    agent.act()
    agent.learn(np.array([0, 2]), np.array([2., -2.]))
    np.testing.assert_array_equal(agent.preferences, np.zeros((2, 3)))
    agent.act()
    pi = agent.probabilities.copy()
    actions = np.array([1, 2])
    agent.learn(actions, np.array([4., 2.]))
    np.testing.assert_array_equal(agent.baseline, [3., 0.])
    expected = .1 * np.array([[1.], [2.]]) * (np.eye(3)[actions] - pi)
    np.testing.assert_allclose(agent.preferences, expected)
    np.testing.assert_allclose(agent.preferences.sum(axis=1), 0, atol=1e-15)

    fixed = GradientAgent(2, 3, METHODS[2], 7)
    fixed.act()
    fixed.learn(actions, np.array([4., 2.]))
    np.testing.assert_array_equal(fixed.baseline, [0., 0.])
    np.testing.assert_allclose(fixed.preferences, .1 * np.array([[4.], [2.]]) * (np.eye(3)[actions] - 1/3))

    a, b = [GradientAgent(16, 10, METHODS[1], 9) for _ in range(2)]
    rewards = np.random.default_rng(11)
    for _ in range(20):
        x, y = a.act(), b.act()
        np.testing.assert_array_equal(x, y)
        r = rewards.normal(size=16)
        a.learn(x, r)
        b.learn(y, r + 4)
        np.testing.assert_allclose(a.preferences, b.preferences, atol=1e-14)
    print("Stable softmax, categorical sampling, simultaneous update, task baselines, timing, and offsets: OK")


if __name__ == "__main__":
    main()
