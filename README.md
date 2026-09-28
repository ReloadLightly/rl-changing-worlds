# Learning in changing worlds

**When does forgetting old experience help a learning agent adapt to a changing environment?**

This small NumPy experiment studies ten-armed bandits from Sutton and Barto,
*Reinforcement Learning: An Introduction*, second edition, Chapter 2. A learner
chooses one of ten actions, receives a noisy reward, and updates its estimate
of that action's value. The aim is to understand the code and the evidence,
including where a simple learning rule fails.

**Measured answer:** forgetting helped when values changed, but the amount
mattered. Of the tested methods, $\alpha=0.1$ minimized total regret in both
changing environments. Sample averages were slightly better just before the
sudden change, and $\alpha=0.5$ adapted fastest immediately afterward. Keeping
more history helps suppress noise; keeping less helps follow a new environment.

## Run it

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python sanity_checks.py
python run_experiments.py
```

The final command runs all three experiments at the requested scale and writes
numerical results and PNG figures to `results/`. It overwrites files with the
same names. Plotting uses Matplotlib's noninteractive backend; no display or GPU
is needed. A small end-to-end check is available:

```bash
python run_experiments.py --smoke
python run_experiments.py --seed 20260929 --output results-another-seed
```

The smoke run uses 32 tasks and 100/200 steps, changes values at step 100, and
saves to `results-smoke/`. Its numbers are not the findings reported here.
`--tasks` changes the number of independent tasks for a full run.

## What the agent learns

An **action** is a choice of arm. Its **true value** $q_t(a)$ is the mean reward
available from that choice at time $t$. The agent cannot see this value. Its
estimate $Q_t(a)$ comes only from rewards it has actually received.

Every estimate and action count starts at zero. An **epsilon-greedy** learner
explores with probability $\epsilon$, choosing uniformly among all ten actions.
Otherwise it chooses an action with the largest estimate, breaking exact ties
uniformly at random. Exploration may also select a greedy action.

After choosing $A_t$ and receiving $R_t$, it increments $N(A_t)$ and updates
only that action:

$$
Q_{t+1}(A_t) = Q_t(A_t) + \eta_t(A_t)\,[R_t-Q_t(A_t)].
$$

The bracket is the prediction error: how surprising was the reward? The step
size determines how much that surprise changes the estimate.

| Learner | Step size after incrementing the count | What it remembers |
| --- | --- | --- |
| Sample average | $\eta_t(a)=1/N(a)$ | Every observed reward for that action has equal weight. |
| Constant step size | $\eta_t(a)=\alpha$ | Recent rewards receive more weight. |

For one action, after $n$ observations, a constant step size gives

$$
Q^{(n)}=(1-\alpha)^n Q^{(0)}
       +\sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i.
$$

Here $i$ counts **visits to this action**, not global time steps. With
$\alpha=0.1$, every new visit retains 90% of the previous estimate. The
approximate memory scale $1/\alpha$ is 100, 10, or 2 visits for the three
constant rates. Unvisited actions retain their estimates: forgetting is not
automatic decay between visits. Persistent exploration is needed to discover
that a neglected action has improved.

For example, rewards 2 and 4 produce the sample average 3. Starting from zero,
$\alpha=0.1$ instead produces estimates 0.2 and then 0.58. That slow start is
part of the method: we use no bias correction, optimistic initialization,
change detector, or reset.

In a stable environment, old observations remain relevant and averaging reduces
reward noise. In a changing environment, old observations can be misleading.
A large constant rate responds quickly but also follows noise more strongly.
These are separate choices from exploration: $\epsilon$ controls which actions
are sampled; $\alpha$ controls what is learned from a sample.

## Experimental design and Chapter 2 connection

All experiments use ten actions and independent reward noise with standard
deviation 1:

$$
R_t = q_t(A_t)+Z_{t,A_t},\qquad Z_{t,a}\sim\mathcal N(0,1).
$$

| Experiment | Environment | Learners | Tasks × steps |
| --- | --- | --- | --- |
| Stationary | Independent initial $q_0(a)\sim\mathcal N(0,1)$, then fixed | Sample averages, $\epsilon=0,0.01,0.1$ | 2,000 × 1,000 |
| Random walk | All $q_0(a)=0$; after every interaction add independent $\mathcal N(0,0.01^2)$ increments to every action | Sample averages and $\alpha=0.01,0.1,0.5$; all $\epsilon=0.1$ | 2,000 × 10,000 |
| **Our extension: sudden change** | Initial independent $\mathcal N(0,1)$ values, randomly permuted among actions at step 5,000, then fixed again | Same four learners as the random walk | 2,000 × 10,000 |

The stationary experiment follows the setup of Figure 2.2. Incremental sample
averages connect to Section 2.4; constant step sizes connect to Section 2.5 and
Exercise 2.5. We add two constant rates to the exercise's comparison. The
sudden permutation is **our extension**, not a claimed textbook reproduction.
See the [book's author page](http://incompleteideas.net/book/the-book-2nd.html)
and [a university-hosted copy](https://www2.imm.dtu.dk/courses/02465/pensum/sutton2018.pdf).

Decisions are indexed from 0. In our extension, decisions 0–4,999 use the
original values; the permutation occurs **before** action selection at decision
5,000. Each task gets its own uniform permutation. Fixed points are allowed:
the best action can remain the best. The multiset of values, and hence its
maximum, is preserved. Agents receive no signal and retain their counts and
estimates across the change.

For the random walk, each interaction uses the current values for both reward
and scoring; only afterward do the values drift. This avoids scoring an action
against a future environment it could not have experienced.

### Shared worlds, separate learners

Within an experiment, every method faces the same initial values, drift path,
permutation, and potential reward $q_t(a)+Z_{t,a}$ for each task, step, and action.
If two methods choose the same action on the same step of the same task, they
receive exactly the same reward. Different actions have independent reward
noise. This pairing makes comparisons less dependent on chance differences in
the environments.

Initial values, reward noise, drift, and permutations have separate random
streams. Each learner also has its own stream, indexed by method identity,
for exploration and tie breaking. The master seed is **20260928**; all derived
integer seeds are saved. Reordering methods cannot change a world's sequence
or a particular learner's results. Independent tasks occupy array rows; there
is no communication or learning across rows. Experiments use distinct streams.

The evaluator knows the true values to measure performance. `Agent.act()` has
no environment input, and `Agent.learn(actions, rewards)` receives only that
learner's selected actions and rewards. No unchosen reward or true value is
passed to a learner.

### Metrics and uncertainty

We report mean reward, the fraction of actions that are optimal, and cumulative
**dynamic pseudo-regret**:

$$
D_T=\sum_{t=0}^{T-1}\left[\max_a q_t(a)-q_t(A_t)\right].
$$

Dynamic means the benchmark may choose a different best action at each step.
Pseudo-regret compares expected rewards, removing the extra noise of comparing
realized reward samples. Each increment is nonnegative. The benchmark is an
evaluator's oracle, not an implementable learner with the same information.

Optimality uses $q_t(A_t)=\max_a q_t(a)$, so **every tied maximizer counts**.
At the random walk's first decision all ten actions are optimal: frequency is
100% and regret is zero, even though the learner has learned nothing.

Confidence intervals use the 2,000 independent tasks as the sampling units:
mean ± $1.96s/\sqrt{2000}$, with sample standard deviation $s$. For a time window
we first compute one reward average, optimal fraction, or regret sum per task.
We never treat correlated time steps as independent replications. Paired
method differences are computed within tasks before constructing intervals.
Intervals are approximate and pointwise, without a multiple-comparison correction.

Reward/frequency figures use nonoverlapping 10-step bins for the stationary
experiment and 100-step bins for the changing experiments. Their intervals
come from per-task bin averages, not smoothed standard errors. Bins do not
straddle the sudden change. Regret is plotted at bin endpoints. Unbinned
per-step means and standard errors are also saved.

## Measured results

The full run completed on **2026-09-28**, using seed **20260928**, Python 3.10.12,
NumPy 1.26.4, and Matplotlib 3.10.9 on Linux/WSL2. Simulation took **2.77 s**
(stationary), **38.87 s** (random walk), and **33.85 s** (sudden change).
The end-to-end run, including saving and plotting, took **80.50 s**. It executed
**166 million agent–environment interactions**. Numerical data, metadata, and
six figures occupy about **8.0 MiB**. These are CPU measurements from this
machine, not a runtime guarantee.

All `±` values below are approximate **95% confidence half-widths for means**.
Optimal frequency is a percentage; its half-width is in percentage points.
Final cumulative regret always covers the **entire run**, independently of the
reward/frequency window. Unrounded results are in the JSON files.

### 1. Stationary: exploration helps discovery

Reward and optimal frequency below use decisions 900–999.

| Sample-average learner | Final-100 reward | Final-100 optimal frequency | Final cumulative regret (1,000 steps) |
| --- | ---: | ---: | ---: |
| $\epsilon=0$ | 1.054 ± 0.028 | 37.6% ± 2.1 | 499.0 ± 25.0 |
| $\epsilon=0.01$ | 1.314 ± 0.028 | 59.0% ± 2.1 | 343.0 ± 18.7 |
| $\epsilon=0.1$ | **1.369 ± 0.025** | **78.9% ± 1.3** | **234.7 ± 5.7** |

![Stationary reward, optimal-action frequency, and cumulative regret](results/stationary.png)

The greedy learner often settles on an inferior action after a few noisy
observations. Exploration lets it revisit alternatives. At this 1,000-step
horizon, $\epsilon=0.1$ did better than $\epsilon=0.01$: their paired final-100
reward difference was **0.056 [0.038, 0.073]**. This does not establish the
ranking at arbitrarily long horizons. Once estimates identify a unique best
action reliably, fixed exploration still selects suboptimal actions: the
optimal-choice probability is $1-\epsilon+\epsilon/10$.

### 2. Random walk: a shrinking step size falls behind

Reward and optimal frequency below use decisions 9,000–9,999; every learner
uses $\epsilon=0.1$.

| Learner | Final-1,000 reward | Final-1,000 optimal frequency | Final cumulative regret (10,000 steps) |
| --- | ---: | ---: | ---: |
| Sample average | 1.049 ± 0.027 | 44.4% ± 1.8 | 3,185.5 ± 62.5 |
| $\alpha=0.01$ | 1.182 ± 0.023 | 54.9% ± 1.7 | 2,629.2 ± 51.1 |
| $\alpha=0.1$ | **1.321 ± 0.023** | **76.5% ± 0.8** | **1,530.8 ± 8.9** |
| $\alpha=0.5$ | 1.204 ± 0.025 | 60.6% ± 0.8 | 2,828.1 ± 9.3 |

![Random-walk reward, optimal-action frequency, and cumulative regret](results/random_walk.png)

Relative to sample averages, $\alpha=0.1$ improved final-window reward by
**0.272 [0.254, 0.289]**, with a paired 95% interval, and reduced total mean
regret by **51.9%**. Its paired total-regret difference was
**−1,654.7 [−1,717.3, −1,592.1]**.

Sample averages increasingly mix obsolete rewards into current estimates.
The constant-rate results show the tradeoff: $\alpha=0.01$ retains too much
history for this drift rate, while $\alpha=0.5$ responds strongly to reward
noise. These results favor $\alpha=0.1$ among the tested rates; they do not
prove it is optimal over all possible rates.

Increasing reward alone is not evidence of improved adaptation: independent
random walks spread out over time, increasing the expected best available
value. Optimal frequency and dynamic regret compare choices with the current
environment and help interpret the reward curve.

### 3. Our extension: abrupt change reveals the cost of old evidence

Reward and optimal frequency below use decisions 9,000–9,999, again with
$\epsilon=0.1$.

| Learner | Final-1,000 reward | Final-1,000 optimal frequency | Final cumulative regret (10,000 steps) |
| --- | ---: | ---: | ---: |
| Sample average | 0.868 ± 0.030 | 36.0% ± 1.9 | 6,137.7 ± 168.4 |
| $\alpha=0.01$ | 1.148 ± 0.024 | 46.8% ± 1.9 | 4,839.6 ± 129.8 |
| $\alpha=0.1$ | **1.358 ± 0.024** | **78.2% ± 0.9** | **2,055.0 ± 27.4** |
| $\alpha=0.5$ | 1.243 ± 0.026 | 60.7% ± 0.9 | 2,974.8 ± 18.2 |

![Our extension: reward, optimal frequency, and cumulative regret around the permutation](results/sudden_change.png)

The timing matters. These are mean rewards in three different windows:

| Learner | Before change: 4,000–4,999 | First 1,000 after: 5,000–5,999 | Entire post-change period: 5,000–9,999 |
| --- | ---: | ---: | ---: |
| Sample average | **1.376** | 0.071 | 0.479 |
| $\alpha=0.01$ | 1.210 | 0.773 | 1.006 |
| $\alpha=0.1$ | 1.356 | 1.175 | **1.320** |
| $\alpha=0.5$ | 1.238 | **1.210** | 1.235 |

Before the change, sample averages beat $\alpha=0.1$ by a paired reward
difference of **0.020 [0.018, 0.022]**. Once the values move, their many old
observations give new evidence very little influence. Even after 5,000 more
interactions, their performance has not returned to its former level.

The largest rate, $\alpha=0.5$, beats $\alpha=0.1$ during the first 1,000
post-change steps by **0.034 [0.023, 0.045]**. It responds quickly, then pays
a continuing cost from noisy estimates. Over the entire post-change period,
$\alpha=0.1$ has the highest reward and beats sample averages by
**0.841 [0.811, 0.871]**. Over all 10,000 steps, it reduces mean regret by
**66.5%** versus sample averages, a paired difference of
**−4,082.8 [−4,233.7, −3,931.9]**.

### Individual trajectories and variation

![Tracking every changing action in the fixed random-walk task](results/random_walk_tracking.png)

![Tracking every action through our sudden-change extension](results/sudden_change_tracking.png)

These figures show the two learners in **task 0, selected before looking at
results**. Black curves are hidden true values; gray and green are estimates
from each learner's own observations. In the sudden-change example, the best
action moves from 6 to 7. The sample-average learner retains a high estimate
for action 6 and a low estimate for action 7; the constant-rate learner revises
both much faster. The green estimate fluctuates when an action is sampled
frequently. A flat estimate for a neglected action need not mean a stable world.

![Distribution of final regret across independent tasks](results/task_variability.png)

Narrow confidence intervals for a mean do not imply similar outcomes on every
task. For sample averages under sudden change, the 10th–90th percentile range
of total regret is **1,747–11,329**. The boxes show this task-to-task spread
separately from uncertainty in the average. One illustrative trajectory cannot
establish the population-level ranking; the 2,000-task comparisons provide that
evidence for these particular environments and horizons.

## Read the code and saved results

Start with [bandits.py](bandits.py): `Agent.act` selects actions and `Agent.learn`
implements the two update rules. `Environment` owns the hidden world;
`simulate` orders interactions and records measurements. NumPy vectorizes
across independent tasks while the time loop remains explicit.
[run_experiments.py](run_experiments.py) sets up the three runs and summarizes
them; [figures.py](figures.py) makes the plots.

[sanity_checks.py](sanity_checks.py) checks only scientific essentials: exact
updates, random ties, tied optima, reward/drift scales, permutation timing and
value preservation, reproducible shared worlds under method reordering, and
regret reconstructed from a logged trajectory. The full runner also checks
finite curves, nondecreasing regret, and correct initial random-walk scoring.

Each `results/<experiment>.npz` can be opened without pickle:

```python
import numpy as np

with np.load("results/random_walk.npz", allow_pickle=False) as data:
    print(data["method_names"])
    # Axes: method, decision step, metric.
    print(data["curve_mean"][:, -1, :])
    # Axes: method, named window, independent task, metric.
    print(data["task_windows"].shape)
```

| Saved item | Meaning |
| --- | --- |
| `method_names`, `metric_names` | Axis labels; metrics are reward, optimal fraction, regret. |
| `curve_mean`, `curve_sem` | Unbinned per-step metrics; regret is cumulative from step 0. |
| `bin_mean`, `bin_sem`, `bin_centers`, `bin_ends` | Plot statistics; regret uses endpoints. |
| `window_names`, `window_bounds`, `task_windows` | Per-task summaries; bounds are `[start, stop)`. Reward and frequency are averages; regret is a **sum within the named window**. |
| `trace_task`, `trace_q_true`, `trace_estimates`, `trace_actions`, `trace_rewards` | Full trajectory for task 0, chosen in advance; estimates recorded before the action. |
| `<experiment>_summary.json` | Means, standard errors, 95% intervals, task SDs, task quantiles, and paired differences against the first listed method. |
| `manifest.json` | Configuration, methods, all seeds, library versions, source SHA-256 hashes, UTC timestamps, and measured runtimes. |

No task × step × action reward cube is retained. Shared potential rewards are
generated one step at a time, and only aggregates, window summaries, and one
task's detailed trajectories are saved. Exact numerical reproduction should
use the NumPy version in the manifest; floating-point and RNG behavior can
differ across software versions. Runtime is machine dependent.

## Food-patch analogy

Think of the ten actions as ten food patches. A visit gives a noisy payoff;
$q_t(a)$ represents expected payoff, not a literal food stock. A sample-average
forager keeps an equally weighted record of every visit. A constant-rate
forager emphasizes recent visits, which can help when patch quality changes.
Occasional exploration revisits patches that used to look unpromising.

This is a bandit analogy. There is no movement, depletion caused by visits,
energy budget, reproduction, competition, or evolving population. It is not
already an artificial life system. The learner stores action-value estimates;
it does not learn a world model or predict environmental transitions. Here the
world changes independently of the learner's choices.

## One worthwhile next experiment

Sweep the drift standard deviation and learning rate together, keeping reward
noise and $\epsilon=0.1$ fixed. Start every condition from shared
$\mathcal N(0,1)$ values so that zero drift provides a meaningful stationary
control. Compare sample averages and several constant rates using paired
final-window reward and regret. This would locate the crossover where retaining
old observations stops helping, and test whether faster environmental change
favors shorter memory. The present three rates and two kinds of change cannot
establish a universally best learning rate.

## Reference implementation and license

We consulted and adapted the epsilon-greedy and incremental-update design of
[Shangtong Zhang's Chapter 2 implementation](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter02/ten_armed_testbed.py).
Its authors' declaration is retained at the top of `bandits.py`; the upstream
MIT license and attribution are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
