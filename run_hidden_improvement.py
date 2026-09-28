"""Will an agent explore when nothing goes wrong? Two matched bandit worlds."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from bandits import Method, make_seeds, simulate
from run_experiments import describe
from run_exploration import METHODS as PREVIOUS_METHODS, write_json


METHODS = PREVIOUS_METHODS + [
    Method("optimistic epsilon=0.01", 0.01, 0.1, initial_estimate=5.0),
    Method("optimistic epsilon=0.1", 0.1, 0.1, initial_estimate=5.0),
]
HORIZONS = (100, 500, 1000, 5000)


def wilson_interval(successes, total):
    """Pointwise 95% binomial intervals, including at zero and full coverage."""
    p = np.asarray(successes) / total
    z2 = 1.96 ** 2
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = 1.96 * np.sqrt(p * (1 - p) / total + z2 / (4 * total ** 2))
    radius /= 1 + z2 / total
    return np.clip(center - radius, 0, 1), np.clip(center + radius, 0, 1)


def unvisited_curve(delays, horizon):
    """S(k): still unvisited AFTER k decisions; delay 0 first changes S(1)."""
    counts = np.bincount(delays[delays >= 0], minlength=horizon)
    remaining = len(delays) - np.r_[0, counts.cumsum()]
    low, high = wilson_interval(remaining, len(delays))
    return remaining / len(delays), low, high


def median_delay(delays, horizon):
    """Empirical 50% crossing; common administrative censoring at the horizon."""
    observed = np.sort(np.where(delays < 0, horizon, delays))
    median = int(observed[int(np.ceil(len(delays) / 2)) - 1])
    return None if median == horizon else median


def task_outcomes(data):
    windows = {name: data["task_windows"][:, w]
               for w, name in enumerate(data["window_names"])}
    start, stop = data["window_bounds"][list(data["window_names"]).index("post_all")]
    post_length = int(stop - start)
    delay = data["target_first_delay"]
    values = {
        "post_regret_per_decision": windows["post_all"][:, :, 2] / post_length,
        "post_reward": windows["post_all"][:, :, 0],
        "first_1000_regret": windows["post_first_1000"][:, :, 2],
        "final_reward": windows["last_1000"][:, :, 0],
        "final_optimal_fraction": windows["last_1000"][:, :, 1],
        "total_regret": windows["all"][:, :, 2],
        "target_pre_visits": data["target_pre_visits"],
        "target_first_1000_frequency": data["target_first_1000_visits"] / 1000,
        "target_last_1000_frequency": data["target_last_1000_visits"] / 1000,
        **{f"visited_within_{h}": ((delay >= 0) & (delay < h)) for h in HORIZONS},
        "unvisited_by_end": delay < 0,
    }
    return np.array(list(values)), np.stack(list(values.values()), axis=2)


def summarize(data):
    names, values = data["outcome_names"], data["task_outcomes"]
    summary = {"methods": {}, "paired_vs_optimistic_greedy": {}}
    for i, method in enumerate(METHODS):
        outcomes = {}
        for k, name in enumerate(names):
            v = values[i, :, k]
            if name.startswith(("visited_", "unvisited_")):
                low, high = wilson_interval(v.sum(), len(v))
                outcomes[name] = {"mean": float(v.mean()), "ci95": [float(low), float(high)]}
            else:
                outcomes[name] = describe(v)
        median = median_delay(data["target_first_delay"][i], 5000)
        summary["methods"][method.name] = dict(
            outcomes=outcomes, median_first_visit_delay=median,
            median_status="not reached within 5000 decisions" if median is None else "reached")
    for i in (6, 7):
        summary["paired_vs_optimistic_greedy"][METHODS[i].name] = {
            name: describe(values[i, :, k] - values[3, :, k])
            for k, name in enumerate(names)
        }
    return summary


def compact(data):
    """Keep task outcomes, plot bins, and task 0; omit redundant window/step arrays."""
    names, values = task_outcomes(data)
    omit = {"curve_mean", "curve_sem", "task_windows", "window_names", "window_bounds"}
    result = {key: value for key, value in data.items() if key not in omit}
    result.update(outcome_names=names, task_outcomes=values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--tasks", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/hidden_improvement"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.tasks < 2 or args.seed < 0:
        parser.error("tasks must be >= 2 and seed nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from hidden_improvement_figures import plot_all

    kinds = ("unchanged", "hidden_improvement")
    if args.plot_only:
        data = {}
        for kind in kinds:
            with np.load(args.output / f"{kind}.npz", allow_pickle=False) as archive:
                data[kind] = dict(archive)
        plot_all(data, args.output)
        return

    started = perf_counter()
    # The same ID and seed dictionaries for BOTH conditions, including learners.
    seeds = make_seeds(args.seed, 7, METHODS)
    manifest = dict(
        started_utc=datetime.now(timezone.utc).isoformat(), master_seed=args.seed,
        experiment_id=7, seeds=seeds, methods=[asdict(m) for m in METHODS],
        python=platform.python_version(), numpy=np.__version__,
        matplotlib=matplotlib.__version__, platform=platform.platform(),
        rng="NumPy default_rng / PCG64; SeedSequence-derived method IDs",
        pairing="identical environment AND per-method learner streams across conditions",
        primary="mean dynamic pseudo-regret per decision, [5000, 10000)",
        censoring="delay -1: no visit in [5000,10000); no observed-only mean reported",
        survival="S(k): fraction unvisited after k decisions; delay d is exposed at k=d+1",
        intervals="task mean +/- 1.96 SEM; Wilson for visit proportions; no multiplicity correction",
        trace_selection="task 0, raster [4500,10000), fixed before inspecting results",
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ("bandits.py", "run_experiments.py", "run_exploration.py",
                                    "run_hidden_improvement.py", "hidden_improvement_figures.py",
                                    "exploration_figures.py")},
        conditions={})
    completed = {}
    for kind in kinds:
        config = dict(kind=kind, tasks=args.tasks, arms=10, steps=10000,
                      bin_width=100, drift_std=0.0, change_step=5000,
                      initialization="normal", reward_std=1.0, improvement_gap=0.5)
        print(f"Running {kind}: {args.tasks:,} tasks × 10,000 decisions", flush=True)
        start = perf_counter()
        full = simulate(config, METHODS, seeds)
        seconds = perf_counter() - start
        assert np.isfinite(full["curve_mean"]).all()
        assert np.all(np.diff(full["curve_mean"][:, :, 2], axis=1) >= -1e-10)
        data = compact(full)
        del full
        if completed:
            before = completed["unchanged"]
            # The worlds are indistinguishable until each learner's first exposure.
            for key in ("initial_q", "target_arms", "target_pre_visits", "target_first_delay"):
                np.testing.assert_array_equal(data[key], before[key])
            np.testing.assert_array_equal(data["bin_mean"][:, :50], before["bin_mean"][:, :50])
        np.savez_compressed(args.output / f"{kind}.npz", **data)
        summary = summarize(data)
        write_json(args.output / f"{kind}_summary.json", summary)
        manifest["conditions"][kind] = dict(config=config, simulation_seconds=seconds)
        write_json(args.output / "manifest.json", manifest)
        completed[kind] = data
        print(f"  simulation: {seconds:.2f} s; post-window regret per decision:", flush=True)
        for name, item in summary["methods"].items():
            print(f"    {name:26s} {item['outcomes']['post_regret_per_decision']['mean']:.4f}", flush=True)
    plot_all(completed, args.output)
    manifest["total_seconds"] = perf_counter() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "manifest.json", manifest)
    print(f"Saved to {args.output}; total {manifest['total_seconds']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
