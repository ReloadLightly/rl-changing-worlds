#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                 #
# 2016 Artem Oboturov(oboturov@gmail.com)                               #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                           #
# Permission given to modify the code as long as you keep this         #
# declaration at the top                                              #
#######################################################################
"""Section 2.9: route experience by observed cue, never by hidden context.

Extends our action-value and gradient learners; see THIRD_PARTY_NOTICES.md.
Each task is an independent array row. Only the active cue is updated.
"""

from dataclasses import dataclass

import numpy as np

from bandits import make_seeds
from gradient_bandits import softmax


@dataclass(frozen=True)
class ContextMethod:
    name: str
    contextual: bool = True
    epsilon: float | None = 0.1
    alpha: float | None = 0.1
    eta: float | None = None


METHODS = [
    ContextMethod("context-blind epsilon=0.1 alpha=0.1", contextual=False),
    ContextMethod("contextual epsilon=0.1 alpha=0.1"),
    ContextMethod("contextual epsilon=0.1 sample average", alpha=None),
    ContextMethod("contextual gradient eta=0.1", epsilon=None, alpha=None, eta=0.1),
]


def context_seeds(master, experiment_id=9, methods=METHODS):
    seeds = make_seeds(master, experiment_id, methods)
    # New roles, local to this experiment; old environment streams are unchanged.
    for role, name in [(4, "true_contexts"), (5, "uninformative_cues")]:
        seeds["environment"][name] = int(np.random.SeedSequence(
            [master, experiment_id, role]).generate_state(1, dtype=np.uint64)[0])
    return seeds


class ContextValueAgent:
    def __init__(self, tasks, arms, method, seed):
        self.method = method
        self.rng = np.random.default_rng(seed)
        self.rows = np.arange(tasks)
        cues = 2 if method.contextual else 1
        self.q = np.zeros((tasks, cues, arms))
        self.counts = np.zeros_like(self.q, dtype=np.int32)

    def act(self, observed_cues):
        self.cues = (observed_cues.copy() if self.method.contextual
                     else np.zeros(len(self.rows), dtype=np.int64))
        values = self.q[self.rows, self.cues]
        ties = values == values.max(axis=1, keepdims=True)
        greedy = np.where(ties, self.rng.random(values.shape), -1).argmax(axis=1)
        explore = self.rng.random(len(self.rows)) < self.method.epsilon
        random_actions = self.rng.integers(values.shape[1], size=len(self.rows))
        return np.where(explore, random_actions, greedy)

    def learn(self, actions, rewards):
        index = (self.rows, self.cues, actions)
        self.counts[index] += 1
        step_size = self.method.alpha
        if step_size is None:
            step_size = 1.0 / self.counts[index]
        self.q[index] += step_size * (rewards - self.q[index])


class ContextGradientAgent:
    """Separate policies and reward baselines; H is not an estimate of reward."""

    def __init__(self, tasks, arms, method, seed):
        self.method = method
        self.rng = np.random.default_rng(seed)
        self.rows = np.arange(tasks)
        self.preferences = np.zeros((tasks, 2, arms))
        self.baselines = np.zeros((tasks, 2))
        self.cue_counts = np.zeros((tasks, 2), dtype=np.int32)

    def act(self, observed_cues):
        self.cues = observed_cues.copy()
        self.probabilities = softmax(self.preferences[self.rows, self.cues])
        cdf = self.probabilities.cumsum(axis=1)
        cdf[:, -1] = 1.0  # Cumulative roundoff only, no exploration floor.
        return (self.rng.random(len(self.rows))[:, None] >= cdf).sum(axis=1)

    def learn(self, actions, rewards):
        index = (self.rows, self.cues)
        self.cue_counts[index] += 1
        self.baselines[index] += (rewards - self.baselines[index]) / self.cue_counts[index]
        advantage = rewards - self.baselines[index]  # Includes the current reward.
        direction = -self.probabilities.copy()  # Cached pre-update policy.
        direction[self.rows, actions] += 1
        self.preferences[index] += self.method.eta * advantage[:, None] * direction


class ContextWorld:
    """Evaluator-owned world. The learners receive only cues and chosen rewards."""

    def __init__(self, tasks, arms, seeds, reward_std=1.0):
        self.rows = np.arange(tasks)
        self.q = np.random.default_rng(seeds["initial"]).normal(size=(tasks, 2, arms))
        self.context_rng = np.random.default_rng(seeds["true_contexts"])
        self.cue_rng = np.random.default_rng(seeds["uninformative_cues"])
        self.reward_rng = np.random.default_rng(seeds["rewards"])
        self.reward_std = reward_std

    def draw(self, informative):
        contexts = self.context_rng.integers(2, size=len(self.rows))
        unrelated_cues = self.cue_rng.integers(2, size=len(self.rows))
        # Consume identical world draws in both conditions, irrespective of behavior.
        noise = self.reward_rng.standard_normal((len(self.rows), self.q.shape[-1]))
        actual_values = self.q[self.rows, contexts]
        observed = contexts.copy() if informative else unrelated_cues
        rewards = actual_values + self.reward_std * noise
        return contexts, observed, actual_values, rewards, noise


def evaluator_values(q, contexts, informative):
    """Conditional action values and both optimums; never passed to a learner."""
    actual = q[np.arange(len(q)), contexts]
    conditional = actual if informative else q.mean(axis=1)
    full_optimum = actual.max(axis=1)
    cue_optimum = conditional.max(axis=1)
    expected_full_optimum = full_optimum if informative else q.max(axis=2).mean(axis=1)
    return conditional, full_optimum, cue_optimum, expected_full_optimum
