# Learning in changing worlds

**When does forgetting old experience help a learning agent adapt to a changing environment?**

This small NumPy experiment studies ten-armed bandits from Sutton and Barto,
*Reinforcement Learning: An Introduction*, second edition, Chapter 2. A learner
chooses one of ten actions, receives a noisy reward, and updates its estimate
of that action's value. The aim is to understand the code and the evidence,
including where a simple learning rule fails.

**First experiments:** forgetting helped when values changed, but the amount
mattered. Of the tested methods, $\alpha=0.1$ minimized total regret in both
changing environments. Sample averages were slightly better just before the
sudden change, and $\alpha=0.5$ had the highest mean reward in the first 1,000
post-change steps. We did not measure a separate recovery time. Keeping
more history helps suppress noise; keeping less helps follow a new environment.

**Memory sweep:** the [15-condition follow-up](#our-memory-sweep-shorter-memory-or-a-noisier-world)
finds that the best tested constant rate increases with stronger drift, while
noise 2 favors smaller rates than noise 1 at the two highest drift levels.
$\alpha=0.1$ wins only 3 of 15 conditions; the pattern is conditional, not universal.

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

## First experiments: measured results

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

## Our memory sweep: shorter memory or a noisier world?

**Question:** how do environmental change and reward noise jointly affect which
learning rate performs best? The three hypotheses, stated before the sweep,
are that faster change may favor larger rates, greater observation noise may
favor smaller rates, and $\alpha=0.1$ may not remain best across conditions.
These are hypotheses to investigate, not assumptions imposed on the results.

**Drift** changes what a patch is expected to provide. Even its average payoff
tomorrow can differ from today's. **Reward noise** makes individual visits
unpredictable even when that expected payoff stays fixed. Averaging repeated
visits can reduce noise, but averaging very old visits can obscure drift.

### Design and commands

This is a new extension of Chapter 2's bandit experiment. **All conditions
start with independent $q_0(a)\sim\mathcal N(0,1)$ values.** This deliberately
differs from the original Exercise 2.5 setup, where all values start equal to
zero. Here zero drift retains meaningful differences between the ten actions.
All learner estimates still start at zero.

After each interaction, every action's value changes independently:

$$
q_{t+1}(a)=q_t(a)+\sigma_d\xi_{t,a},\qquad
R_t=q_t(A_t)+\sigma_r Z_{t,A_t},\qquad
\xi_{t,a},Z_{t,a}\sim\mathcal N(0,1).
$$

| Setting | Tested values |
| --- | --- |
| Drift standard deviation $\sigma_d$ | 0, 0.001, 0.003, 0.01, 0.03 |
| Reward standard deviation $\sigma_r$ | 0.5, 1, 2 |
| Learners | Sample averages; constant $\alpha=0.003,0.01,0.03,0.1,0.3,0.5$ |
| Exploration | $\epsilon=0.1$ for every learner |
| Replications and horizon | 2,000 independent tasks × 10,000 decisions in each of 15 conditions |

The same initial values, standard-normal drift draws, and standard-normal
potential reward draws are reused **across all conditions**, scaled by the
specified standard deviations. Methods within a condition therefore encounter
identical potential rewards for identical choices. The sweep restarts the
same separate stream for each method in every condition, too; no learner
observes another learner's data. Drift/noise settings, true values, and unchosen
rewards are available only to the environment and evaluator.

The master seed remains 20260928, with a new experiment ID of 3. Existing
method seed IDs are preserved, and the extra rates have explicit new IDs;
Python's randomized `hash()` is not used. Changing the reward standard
deviation now actually scales reward generation. The original experiments all
used standard deviation 1 and retain their original defaults and saved results.

```bash
source .venv/bin/activate
python memory_sweep_checks.py
python run_memory_sweep.py
# If interrupted, reuse completed conditions with matching configuration/seeds/code:
python run_memory_sweep.py --resume
# Rebuild tables and figures using saved completed conditions:
python run_memory_sweep.py --plot-only
```

The new results live in `results/memory_sweep/`. Each completed condition saves
a compressed `.npz` and a JSON record with configuration, seeds, source hashes,
timing, and summaries. The manifest records progress after each condition.
`--resume` rejects mismatched saved conditions; use `--output` for a separate
run. No original experiment needs to be rerun for this sweep.

### Outcome, weighting, and uncertainty

The **primary outcome** is each task's mean dynamic pseudo-regret per decision
over decisions 9,000–9,999:

$$
L=\frac{1}{1000}\sum_{t=9000}^{9999}
\left[\max_a q_t(a)-q_t(A_t)\right].
$$

Lower is better. Secondary outcomes are total 10,000-step regret, final-window
mean reward, and final-window optimal-action frequency. For each outcome we
compute method-minus-sample-average differences within the same task, then
average these paired differences across 2,000 tasks. Negative regret differences
favor the constant-rate method; positive reward/frequency differences favor it.

Constant $\alpha$ weights a reward from $k$ visits ago by
$\alpha(1-\alpha)^k$. Larger rates emphasize newer evidence and transmit more
reward noise. The approximate $1/\alpha$ memory scales here are 333, 100, 33,
10, 3.3, and 2 **visits to an action**. They are not global environment steps:
an action visited rarely can retain an old estimate for a long time. The
initial estimate also retains weight $(1-\alpha)^n$ after $n$ visits, which can
matter at finite horizons for the smallest rates.

Intervals are mean ± 1.96 across-task standard errors. We also compare the
empirical winner with the runner-up and other methods using paired differences.
These are **exploratory, pointwise comparisons**: selecting the winner after
seeing the same data and making many comparisons is not accounted for by the
ordinary 95% intervals. An interval including zero does not establish equality,
and the lowest tested mean does not identify a universally optimal rate.
Overlapping intervals for separate means do not determine whether a paired
difference is distinguishable from zero. Conditions reuse tasks, so they are
not 15 independent replications of a trend.

The heatmap colors compare methods **within the same environment**, using
excess primary regret over that row's best mean; cell labels show absolute
primary regret and its interval. This matters because faster random walks
also spread action values farther apart. Starting at variance 1 gives marginal
variance $1+t\sigma_d^2$. Larger gaps change the attainable reward, the cost
of exploration, and the scale of regret. Raw comparisons across drift levels
therefore do not isolate adaptation difficulty alone.

The three learning-curve conditions were selected in advance: zero drift with
noise 1, drift 0.01 with noise 1, and drift 0.01 with noise 2. Curves use the same
100-step task averages and pointwise intervals as the first experiments.

### Measured sweep results

The full sweep completed on **2026-09-28** with **2.1 billion interactions**.
Simulation took **894.12 s**; saving, tables, and plotting brought the full run
to **897.95 s (14 min 58 s)**. It used Python 3.10.12, NumPy 1.26.4, and
Matplotlib 3.10.9 on the same Linux/WSL2 CPU environment as the first experiments.
The new data, metadata, and five figures occupy **7.5 MiB**. All 13 original
result files were preserved byte-for-byte; the original experiments were not rerun.

The table shows the **lowest measured primary outcome** among the seven
learners, not a universally optimal setting. SA means sample averages.

| Drift standard deviation | Reward noise 0.5 | Reward noise 1 | Reward noise 2 |
| --- | --- | --- | --- |
| 0 | SA† | SA | SA |
| 0.001 | $\alpha=0.03$ | SA | SA |
| 0.003 | $\alpha=0.03$ | $\alpha=0.03$ | $\alpha=0.03$‡ |
| 0.01 | $\alpha=0.1$ | $\alpha=0.1$ | $\alpha=0.03$ |
| 0.03 | $\alpha=0.3$† | $\alpha=0.3$ | $\alpha=0.1$ |

† The paired 95% interval against the runner-up includes zero. ‡ The interval
only just excludes zero. Details appear below; none is adjusted for selection
or multiple comparisons.

![Primary outcome by drift and method with reward noise 0.5](results/memory_sweep/heatmap_noise_0.5.png)

![Primary outcome by drift and method with reward noise 1](results/memory_sweep/heatmap_noise_1.png)

![Primary outcome by drift and method with reward noise 2](results/memory_sweep/heatmap_noise_2.png)

![Best tested methods and constant learning rates across drift and reward noise](results/memory_sweep/best_tested_rates.png)

**Hypothesis 1: supported directionally.** Among constant-rate learners, noise
0.5 and noise 1 both give the sequence **0.03, 0.03, 0.03, 0.1, 0.3** as drift
increases. Noise 2 gives **0.03, 0.03, 0.03, 0.03, 0.1**. Faster change favored
larger rates at the upper end of the tested grid, but not at every increase in
drift. Sample averages won the three stationary conditions and two of the
smallest-drift conditions. Because drift also enlarges value gaps, this is
evidence for these random-walk environments, not an isolated causal law about
change speed alone.

**Hypothesis 2: supported at higher drift, not uniformly.** At drift 0.01,
increasing noise from 1 to 2 changes the winner from $\alpha=0.1$ to
$\alpha=0.03$. At noise 2, their mean regrets are **0.2825** and **0.2483** per
decision: the smaller rate's paired difference is
**−0.0342 [−0.0384, −0.0299]**. At drift 0.03 the winner changes from 0.3 to
0.1; with noise 2, rate 0.1 beats rate 0.3 by
**−0.0472 [−0.0531, −0.0413]**. These comparisons are within each environment.
However, **no best constant rate changed when noise rose from 0.5 to 1**, and
the lowest three drift levels selected 0.03 at every noise level. The stronger
claim that every increase in noise should lower the best tested rate is not
supported by this sweep.

**Hypothesis 3: supported.** $\alpha=0.1$ wins **3/15** conditions by the primary
outcome; sample averages win 5, rate 0.03 wins 5, and rate 0.3 wins 2. The first
experiments identified a useful setting for their comparisons, not a general
default that wins everywhere. These counts describe this grid and are not
independent replications or a statistical test of the hypotheses.

#### Close comparisons

All differences here are **first method minus second method**, in final-window
regret per decision; positive values favor the second method.

| Drift / noise | Comparison | Paired mean difference [95% interval] |
| --- | --- | ---: |
| 0 / 0.5 | $\alpha=0.03$ − SA | 0.00031 [−0.00084, 0.00146] |
| 0.03 / 0.5 | $\alpha=0.5$ − $\alpha=0.3$ | 0.00035 [−0.00329, 0.00398] |
| 0.003 / 2 | SA − $\alpha=0.03$ | 0.00434 [0.00013, 0.00855] |
| 0 / 2 | $\alpha=0.01$ − $\alpha=0.03$ (constants only) | 0.00156 [−0.00253, 0.00566] |
| 0.001 / 2 | $\alpha=0.01$ − $\alpha=0.03$ (constants only) | 0.00265 [−0.00157, 0.00687] |

The first two point-estimate winners are not clearly separated from their
runner-up by these intervals. The drift-0.003/noise-2 comparison has only
marginal unadjusted evidence; it should not be treated as a firm ranking after
searching this many comparisons. The last two rows qualify the constant-rate
panel: sample averages win overall in both conditions, while the ranking
between the two best constants is uncertain. Intervals containing zero are
not evidence that two methods are equivalent.

#### Preselected learning curves and secondary outcomes

![Learning curves for the three conditions chosen before running](results/memory_sweep/learning_curves.png)

The following are the sample-average baseline and the best tested constant
rate in each preselected condition. Primary-outcome `±` values are 95% confidence
half-widths. The other columns are means; intervals for all four outcomes and
every learner are in `summary.csv` and the condition JSON files.

| Drift / noise | Learner | Final regret/decision | Total regret | Final reward | Final optimal frequency |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 / 1 | Sample average | 0.1561 ± 0.0023 | 1,659.5 | 1.3850 | 86.6% |
| 0 / 1 | $\alpha=0.03$ | 0.1592 ± 0.0022 | 2,146.6 | 1.3810 | 84.6% |
| 0.01 / 1 | Sample average | 0.4052 ± 0.0144 | 3,113.7 | 1.7492 | 59.1% |
| 0.01 / 1 | $\alpha=0.1$ | 0.2358 ± 0.0031 | 2,229.6 | 1.9200 | 80.9% |
| 0.01 / 2 | Sample average | 0.4021 ± 0.0145 | 3,288.5 | 1.7543 | 59.7% |
| 0.01 / 2 | $\alpha=0.03$ | 0.2483 ± 0.0040 | 2,592.8 | 1.9083 | 77.4% |

At drift 0.01/noise 1, rate 0.1 reduces the primary outcome relative to sample
averages by **−0.1694 [−0.1840, −0.1548]**. At drift 0.01/noise 2, rate 0.03
reduces it by **−0.1538 [−0.1683, −0.1392]**. Under zero drift/noise 1, sample
averages instead beat rate 0.03 by **0.00306 [0.00166, 0.00446]** in regret per
decision. The distinction between retaining useful evidence and retaining
outdated evidence changes with the environment.

The smallest constant rate, **0.003, never won**, even with zero drift. Its
learning curves show slow improvement within this horizon. A small rate also
retains the zero initial estimate and updates an infrequently visited action
slowly; it is not the same as a well-established estimate with low variance.
Sample averages take a full first update, then gradually reduce their step
size. This experiment does not separate initialization effects from later
tracking error, so the finite-horizon result does not show that long memory
is intrinsically bad in a stationary environment.

### Compact data

[summary.csv](results/memory_sweep/summary.csv) contains all 105 condition–method
rows, with all four outcomes, 95% intervals, and paired differences against
sample averages. Each condition's JSON also records the best tested method,
best tested constant rate, runner-up comparisons, task SDs, and task quantiles.
Each `.npz` contains `task_outcomes` with axes **method × task × outcome**:
final regret per decision, total regret, final reward, final optimal fraction.
`outcome_names` and `method_names` label these axes. The other arrays contain
100-step bin means/standard errors and plotting coordinates. We retain no
large task-by-time-by-action cube and no full per-task trajectories for the sweep.

## Read the code and saved results

Start with [bandits.py](bandits.py): `Agent.act` selects actions and `Agent.learn`
implements the two update rules. `Environment` owns the hidden world;
`simulate` orders interactions and records measurements. NumPy vectorizes
across independent tasks while the time loop remains explicit.
[run_experiments.py](run_experiments.py) sets up the three runs and summarizes
them; [figures.py](figures.py) makes the plots.

[run_memory_sweep.py](run_memory_sweep.py) adds the drift–noise sweep using the
same `simulate` loop. [memory_sweep_figures.py](memory_sweep_figures.py) plots it.
The sweep saves separate results under `results/memory_sweep/`; the original
result files remain unchanged. Its brief checks are in
[memory_sweep_checks.py](memory_sweep_checks.py).

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

## One next scientific question

Do faster changes favor larger learning rates when the spread of action values
is held fixed? A follow-up could use
$q_{t+1}(a)=\rho q_t(a)+\sqrt{1-\rho^2}\,\xi_{t,a}$, initialized from
$\mathcal N(0,1)$. This keeps marginal variance at 1 while varying temporal
persistence through $\rho$. Fix reward noise and exploration, then repeat the
learning-rate comparison. It would help separate the effect of change speed
from the growing action-value gaps in an unrestricted random walk.

## Reference implementation and license

We consulted and adapted the epsilon-greedy and incremental-update design of
[Shangtong Zhang's Chapter 2 implementation](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter02/ten_armed_testbed.py).
Its authors' declaration is retained at the top of `bandits.py`; the upstream
MIT license and attribution are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
