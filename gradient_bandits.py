#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                 #
# 2016 Artem Oboturov(oboturov@gmail.com)                               #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                           #
# Permission given to modify the code as long as you keep this         #
# declaration at the top                                              #
#######################################################################
"""Section 2.8: learn preferences, not expected-reward estimates.

Adapted from the gradient branch of Shangtong Zhang's ten_armed_testbed.py;
see THIRD_PARTY_NOTICES.md. Tasks are independent array rows. The learner
receives only its own actions and rewards, never the evaluator's target.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GradientMethod:
    name: str
    eta: float
    running_baseline: bool


METHODS = [
    GradientMethod("gradient eta=0.1 running baseline", 0.1, True),
    GradientMethod("gradient eta=0.4 running baseline", 0.4, True),
    GradientMethod("gradient eta=0.1 zero baseline", 0.1, False),
    GradientMethod("gradient eta=0.4 zero baseline", 0.4, False),
]


def softmax(preferences):
    shifted = preferences - preferences.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


class GradientAgent:
    """A softmax policy and, optionally, a separate mean reward for every task."""

    def __init__(self, tasks, arms, method, seed):
        self.method = method
        self.rng = np.random.default_rng(seed)
        self.preferences = np.zeros((tasks, arms))
        self.baseline = np.zeros(tasks)
        self.time = 0
        self.rows = np.arange(tasks)

    def act(self):
        # Retain exactly the probabilities that generated this decision.
        self.probabilities = softmax(self.preferences)
        cdf = self.probabilities.cumsum(axis=1)
        cdf[:, -1] = 1.0  # Fix cumulative roundoff, not a probability floor.
        draws = self.rng.random(len(self.rows))
        return (draws[:, None] >= cdf).sum(axis=1)

    def learn(self, actions, rewards):
        self.time += 1
        if self.method.running_baseline:
            # Textbook convention: include THIS reward before updating H.
            self.baseline += (rewards - self.baseline) / self.time
        advantage = rewards - self.baseline
        direction = -self.probabilities.copy()
        direction[self.rows, actions] += 1.0
        # Simultaneous update of ALL preferences using the pre-update policy.
        self.preferences += self.method.eta * advantage[:, None] * direction
