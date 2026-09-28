#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                 #
# 2016 Artem Oboturov(oboturov@gmail.com)                               #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                           #
# Permission given to modify the code as long as you keep this         #
# declaration at the top                                              #
#######################################################################
"""Chapter 2 epsilon-greedy bandits, vectorized over independent tasks.

Adapted conceptually from Shangtong Zhang's ten_armed_testbed.py; see
THIRD_PARTY_NOTICES.md. Environmental dynamics and measurement are separate
from the learner: a learner sees only its own chosen actions and rewards.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Method:
    name: str
    epsilon: float
    alpha: float | None = None  # None means sample averages, step size 1/N(a).


STATIONARY_METHODS = [Method(f"epsilon={e:g}", e) for e in (0, 0.01, 0.1)]
TRACKING_METHODS = [Method("sample average", 0.1)] + [
    Method(f"alpha={a:g}", 0.1, a) for a in (0.01, 0.1, 0.5)
]


class Agent:
    """One learner per task, using only estimates, counts, and its own RNG."""

    def __init__(self, tasks, arms, method, seed):
        self.method = method
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((tasks, arms))
        self.counts = np.zeros((tasks, arms), dtype=np.int64)
        self.rows = np.arange(tasks)

    def act(self):
        # Independent random scores select uniformly among *all* exact ties.
        ties = self.q == self.q.max(axis=1, keepdims=True)
        scores = self.rng.random(self.q.shape)
        greedy = np.where(ties, scores, -1).argmax(axis=1)
        explore = self.rng.random(len(self.rows)) < self.method.epsilon
        random_actions = self.rng.integers(self.q.shape[1], size=len(self.rows))
        return np.where(explore, random_actions, greedy)

    def learn(self, actions, rewards):
        self.counts[self.rows, actions] += 1
        step_size = self.method.alpha
        if step_size is None:
            step_size = 1.0 / self.counts[self.rows, actions]
        old = self.q[self.rows, actions]
        self.q[self.rows, actions] = old + step_size * (rewards - old)


class Environment:
    """Shared potential outcomes; no agent can access this object."""

    def __init__(self, kind, tasks, arms, seeds, drift_std=0.01, change_step=5000):
        self.kind = kind
        self.drift_std = drift_std
        self.change_step = change_step
        self.reward_rng = np.random.default_rng(seeds["rewards"])
        self.drift_rng = np.random.default_rng(seeds["drift"])
        initial_rng = np.random.default_rng(seeds["initial"])
        self.q = (np.zeros((tasks, arms)) if kind == "random_walk"
                  else initial_rng.normal(size=(tasks, arms)))
        permutation_rng = np.random.default_rng(seeds["permutation"])
        # A separate, uniform permutation per task; fixed points are allowed.
        self.permutation = permutation_rng.permuted(
            np.broadcast_to(np.arange(arms), (tasks, arms)), axis=1
        )

    def before_action(self, step):
        if self.kind == "sudden_change" and step == self.change_step:
            self.q = np.take_along_axis(self.q, self.permutation, axis=1)

    def potential_rewards(self):
        # One independent noise draw per (task, step, action), shared by methods.
        return self.q + self.reward_rng.normal(size=self.q.shape)

    def after_interaction(self):
        if self.kind == "random_walk":
            self.q += self.drift_rng.normal(0, self.drift_std, self.q.shape)


def make_seeds(master, experiment_id, methods):
    """Stable named streams: adding/reordering methods cannot change the world."""
    def seed(role):
        sequence = np.random.SeedSequence([master, experiment_id, role])
        return int(sequence.generate_state(1, dtype=np.uint64)[0])

    environment = {name: seed(i) for i, name in enumerate(
        ("initial", "rewards", "drift", "permutation")
    )}
    # Explicit IDs keep streams tied to methods rather than list positions.
    ids = {"epsilon=0": 100, "epsilon=0.01": 101, "epsilon=0.1": 102,
           "sample average": 200, "alpha=0.01": 201, "alpha=0.1": 202,
           "alpha=0.5": 203}
    return {"environment": environment,
            "agents": {m.name: seed(ids[m.name]) for m in methods}}


def action_metrics(q_true, actions):
    chosen = q_true[np.arange(len(actions)), actions]
    best = q_true.max(axis=1)
    # All tied maximizers count, including every action at random-walk step 0.
    return chosen == best, best - chosen


def mean_sem(values, axis=0):
    return values.mean(axis=axis), values.std(axis=axis, ddof=1) / np.sqrt(
        values.shape[axis]
    )


def simulate(config, methods, seeds):
    """Run all methods on each shared world; retain aggregates and task 0.

    Metrics order: reward, optimal-action indicator, cumulative pseudo-regret.
    Plot bins average reward/optimal indicators *within each task first*;
    cumulative regret is sampled at the bin end, never averaged over time.
    """
    tasks, steps, arms = (config[k] for k in ("tasks", "steps", "arms"))
    width = config["bin_width"]
    env = Environment(config["kind"], tasks, arms, seeds["environment"],
                      config["drift_std"], config["change_step"])
    agents = [Agent(tasks, arms, m, seeds["agents"][m.name]) for m in methods]
    rows = np.arange(tasks)
    shape = (len(methods), steps, 3)
    curve_mean, curve_sem = np.zeros(shape), np.zeros(shape)
    bin_shape = (len(methods), (steps + width - 1) // width, 3)
    bin_mean, bin_sem = np.zeros(bin_shape), np.zeros(bin_shape)
    bin_totals = np.zeros((len(methods), tasks, 2))
    cumulative = np.zeros((len(methods), tasks))

    windows = {"all": (0, steps), "last_100": (max(0, steps - 100), steps),
               "last_1000": (max(0, steps - 1000), steps)}
    if config["kind"] == "sudden_change":
        change = config["change_step"]
        windows.update(pre_last_1000=(max(0, change - 1000), change),
                       post_first_1000=(change, min(steps, change + 1000)),
                       post_all=(change, steps))
    window_totals = np.zeros((len(methods), len(windows), tasks, 3))
    trace_q_true = np.zeros((steps, arms))
    trace_estimates = np.zeros((len(methods), steps, arms))
    trace_actions = np.zeros((len(methods), steps), dtype=np.int16)
    trace_rewards = np.zeros((len(methods), steps))

    for t in range(steps):
        env.before_action(t)
        rewards_for_all_actions = env.potential_rewards()
        trace_q_true[t] = env.q[0]
        active_windows = [w for w, (start, stop) in enumerate(windows.values())
                          if start <= t < stop]
        bin_end = (t + 1) % width == 0 or t == steps - 1
        for i, agent in enumerate(agents):
            trace_estimates[i, t] = agent.q[0]  # Before observing this reward.
            actions = agent.act()
            rewards = rewards_for_all_actions[rows, actions]
            optimal, regret = action_metrics(env.q, actions)
            cumulative[i] += regret
            values = np.column_stack((rewards, optimal, cumulative[i]))
            curve_mean[i, t], curve_sem[i, t] = mean_sem(values)
            bin_totals[i] += values[:, :2]
            increments = np.column_stack((rewards, optimal, regret))
            for w in active_windows:
                window_totals[i, w] += increments
            if bin_end:
                bin_length = t % width + 1
                binned = np.column_stack((bin_totals[i] / bin_length, cumulative[i]))
                bin_mean[i, t // width], bin_sem[i, t // width] = mean_sem(binned)
                bin_totals[i] = 0
            trace_actions[i, t] = actions[0]
            trace_rewards[i, t] = rewards[0]
            agent.learn(actions, rewards)
        # Drift follows reward, scoring, and learning, including at the last step.
        env.after_interaction()
        if (t + 1) % 1000 == 0:
            print(f"  {config['kind']}: {t + 1:,}/{steps:,} steps", flush=True)

    for w, (start, stop) in enumerate(windows.values()):
        window_totals[:, w, :, :2] /= stop - start
    starts = np.arange(0, steps, width)
    ends = np.minimum(starts + width, steps) - 1
    return dict(
        method_names=np.array([m.name for m in methods]),
        metric_names=np.array(["reward", "optimal_fraction", "cumulative_regret"]),
        curve_mean=curve_mean, curve_sem=curve_sem,
        bin_mean=bin_mean, bin_sem=bin_sem, bin_centers=(starts + ends) / 2,
        bin_ends=ends, window_names=np.array(list(windows)),
        window_bounds=np.array(list(windows.values())), task_windows=window_totals,
        trace_task=np.array(0), trace_q_true=trace_q_true,
        trace_estimates=trace_estimates, trace_actions=trace_actions,
        trace_rewards=trace_rewards,
    )
