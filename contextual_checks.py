"""Brief scientific checks: cue routing, local clocks, and benchmark conditioning."""

import numpy as np

from contextual_bandits import (METHODS, ContextGradientAgent, ContextValueAgent,
                                ContextWorld, context_seeds, evaluator_values)


def main():
    rows, cues, actions = np.arange(2), np.array([0, 1]), np.array([1, 2])
    value = ContextValueAgent(2, 3, METHODS[2], 7)
    for c, r in [(cues, [2., 4.]), (1-cues, [10., 20.]), (cues, [4., 8.])]:
        value.act(c)
        value.learn(actions, np.array(r))
    np.testing.assert_array_equal(value.q[rows, cues, actions], [3., 6.])
    np.testing.assert_array_equal(value.q[rows, 1-cues, actions], [10., 20.])
    np.testing.assert_array_equal(value.counts[rows, cues, actions], [2, 2])
    constant = ContextValueAgent(2, 3, METHODS[1], 7)
    constant.act(cues)
    constant.learn(actions, np.array([2., 4.]))
    np.testing.assert_allclose(constant.q[rows, cues, actions], [.2, .4])
    np.testing.assert_array_equal(constant.q[rows, 1-cues], np.zeros((2, 3)))

    gradient = ContextGradientAgent(2, 3, METHODS[3], 7)
    for c, r in [(cues, [2., 4.]), (1-cues, [10., 20.])]:
        gradient.act(c)
        gradient.learn(actions, np.array(r))
    np.testing.assert_array_equal(gradient.preferences, np.zeros((2, 2, 3)))
    gradient.preferences[rows, cues] = [-1., 0., 1.]
    before = gradient.preferences.copy()
    gradient.act(cues)
    pi = gradient.probabilities.copy()
    gradient.learn(actions, np.array([4., 8.]))
    np.testing.assert_array_equal(gradient.baselines[rows, cues], [3., 6.])
    np.testing.assert_array_equal(gradient.baselines[rows, 1-cues], [10., 20.])
    np.testing.assert_array_equal(gradient.cue_counts[rows, cues], [2, 2])
    expected = before[rows, cues] + .1 * np.array([[1.], [2.]]) * (np.eye(3)[actions] - pi)
    np.testing.assert_allclose(gradient.preferences[rows, cues], expected)
    np.testing.assert_array_equal(gradient.preferences[rows, 1-cues], before[rows, 1-cues])

    a, b = [ContextValueAgent(2, 3, METHODS[0], 9) for _ in range(2)]
    for _ in range(5):
        x, y = a.act(cues), b.act(1-cues)
        np.testing.assert_array_equal(x, y)
        a.learn(x, np.array([2., 4.]))
        b.learn(y, np.array([2., 4.]))
    np.testing.assert_array_equal(a.q, b.q)

    q = np.array([[[0., 2.], [4., 0.]]])
    conditional, full, cue, expected_full = evaluator_values(q, np.array([0]), False)
    np.testing.assert_array_equal(conditional, [[2., 1.]])
    np.testing.assert_array_equal([full, cue, expected_full], [[2.], [2.], [3.]])
    # Choosing arm 0: cue regret 0, realized full regret 2, expected full regret 1.
    assert cue[0] - conditional[0, 0] == 0
    seeds = context_seeds(20260928)
    worlds = [ContextWorld(32, 10, seeds["environment"]) for _ in range(2)]
    for _ in range(4):
        x, y = worlds[0].draw(True), worlds[1].draw(False)
        for k in (0, 2, 3, 4):
            np.testing.assert_array_equal(x[k], y[k])
        np.testing.assert_array_equal(x[0], x[1])
    assert context_seeds(20260928, methods=METHODS[::-1]) == seeds
    print("Cue routing, value updates, cue-local baseline timing, cached gradient, blind invariance, and benchmarks: OK")


if __name__ == "__main__":
    main()
