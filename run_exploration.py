"""Chapter 2 optimism and UCB in stationary, drifting, and permuted worlds."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from bandits import Method, make_seeds, simulate
from run_experiments import describe


# Numbers match the research design; old method identities keep their old seeds.
METHODS = [
    Method("sample average", 0.1),
    Method("alpha=0.1", 0.1, 0.1),
    Method("greedy alpha=0.1", 0.0, 0.1),
    Method("optimistic greedy", 0.0, 0.1, initial_estimate=5.0),
    Method("UCB sample average", 0.0, ucb_c=2.0),
    Method("UCB alpha=0.1 (variant)", 0.0, 0.1, ucb_c=2.0),
]
PAIRS = [(4, 3), (5, 1), (6, 2), (6, 5)]  # One-based; first minus second.


def summarize(data):
    """Paired differences use one observation per independent task, not step."""
    summary = {"windows": {}, "paired_differences": {}}
    metrics = ["reward", "optimal_fraction", "regret", "regret_per_decision"]
    for w, window in enumerate(data["window_names"]):
        start, stop = data["window_bounds"][w]
        values = data["task_windows"][:, w]
        values = np.concatenate((values, values[:, :, 2:3] / (stop - start)), axis=2)
        summary["windows"][window] = {
            method.name: {metric: describe(values[i, :, k])
                          for k, metric in enumerate(metrics)}
            for i, method in enumerate(METHODS)
        }
        summary["paired_differences"][window] = {
            f"{a}_minus_{b}": {
                "methods": [METHODS[a - 1].name, METHODS[b - 1].name],
                **{metric: describe(values[a - 1, :, k] - values[b - 1, :, k])
                   for k, metric in enumerate(metrics)},
            } for a, b in PAIRS
        }
    return summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--tasks", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/exploration"))
    parser.add_argument("--plot-only", action="store_true",
                        help="rebuild figures from saved arrays; do not simulate")
    args = parser.parse_args()
    if args.tasks < 2 or args.seed < 0:
        parser.error("tasks must be >= 2 and seed must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from exploration_figures import plot_curves, plot_textbook_pairs, plot_action_raster

    if args.plot_only:
        manifest = json.loads((args.output / "manifest.json").read_text())
        for kind, record in manifest["experiments"].items():
            with np.load(args.output / f"{kind}.npz", allow_pickle=False) as data:
                plot_curves(data, record["config"], args.output)
                if kind == "stationary":
                    plot_textbook_pairs(data, args.output)
                if kind == "sudden_change":
                    plot_action_raster(data, manifest["raster_window"], args.output)
        return

    started = perf_counter()
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": args.seed, "python": platform.python_version(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "rng": "NumPy default_rng / PCG64",
        "methods": [asdict(method) for method in METHODS], "pairs": PAIRS,
        "step_convention": "zero-based; pre-decision counts; change before t=5000",
        "trace_selection": "task 0, chosen before inspecting results",
        "raster_window": [4500, 6500],  # Fixed in advance, [start, stop).
        "ci": "mean +/- 1.96 * task sample SD / sqrt(tasks); paired within tasks",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("bandits.py", "run_experiments.py", "run_exploration.py",
                         "exploration_figures.py")
        },
        "experiments": {},
    }
    # New experiment IDs leave the earlier experiments and sweep streams intact.
    for experiment_id, kind in enumerate(
            ("stationary", "random_walk", "sudden_change"), start=4):
        stationary = kind == "stationary"
        config = dict(kind=kind, tasks=args.tasks, arms=10,
                      steps=1000 if stationary else 10000,
                      bin_width=10 if stationary else 100, drift_std=0.01,
                      reward_std=1.0, change_step=5000,
                      initialization="zeros" if kind == "random_walk" else "normal")
        seeds = make_seeds(args.seed, experiment_id, METHODS)
        print(f"Running {kind}: {args.tasks:,} tasks, {config['steps']:,} decisions",
              flush=True)
        simulation_start = perf_counter()
        data = simulate(config, METHODS, seeds)
        seconds = perf_counter() - simulation_start
        assert np.isfinite(data["curve_mean"]).all()
        assert np.all(np.diff(data["curve_mean"][:, :, 2], axis=1) >= -1e-10)
        if kind == "random_walk":
            assert np.all(data["curve_mean"][:, 0, 1] == 1)
            assert np.all(data["curve_mean"][:, 0, 2] == 0)
        # Save each complete experiment before plotting or starting another one.
        np.savez_compressed(args.output / f"{kind}.npz", **data)
        summary = summarize(data)
        write_json(args.output / f"{kind}_summary.json", summary)
        manifest["experiments"][kind] = dict(
            experiment_id=experiment_id, config=config, seeds=seeds,
            simulation_seconds=seconds)
        write_json(args.output / "manifest.json", manifest)
        plot_curves(data, config, args.output)
        if stationary:
            plot_textbook_pairs(data, args.output)
        if kind == "sudden_change":
            plot_action_raster(data, manifest["raster_window"], args.output)
        print(f"  simulation: {seconds:.2f} s; mean total regret:", flush=True)
        for name, result in summary["windows"]["all"].items():
            print(f"    {name:25s} {result['regret']['mean']:.2f}", flush=True)
    manifest["total_seconds"] = perf_counter() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "manifest.json", manifest)
    print(f"Saved to {args.output}; total {manifest['total_seconds']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
