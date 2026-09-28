"""Run the three Chapter 2 experiments and save compact, auditable results."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from bandits import STATIONARY_METHODS, TRACKING_METHODS, make_seeds, mean_sem, simulate


def describe(values):
    mean, sem = mean_sem(values)
    return {"mean": float(mean), "sem": float(sem),
            "ci95": [float(mean - 1.96 * sem), float(mean + 1.96 * sem)],
            "task_sd": float(values.std(ddof=1)),
            "task_quantiles_10_50_90": np.quantile(values, [0.1, 0.5, 0.9]).tolist()}


def summarize(data):
    """Use tasks as independent units, including for paired method differences."""
    summary = {"windows": {}, "paired_differences_vs_first_method": {}}
    for w, window in enumerate(data["window_names"]):
        summary["windows"][window] = {}
        summary["paired_differences_vs_first_method"][window] = {}
        for m, method in enumerate(data["method_names"]):
            values = data["task_windows"][m, w]
            summary["windows"][window][method] = {
                metric: describe(values[:, k])
                for k, metric in enumerate(data["metric_names"])
            }
            if m:
                difference = values - data["task_windows"][0, w]
                summary["paired_differences_vs_first_method"][window][method] = {
                    metric: describe(difference[:, k])
                    for k, metric in enumerate(data["metric_names"])
                }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--tasks", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--smoke", action="store_true",
                        help="32 tasks, 100/200 steps; defaults to results-smoke")
    args = parser.parse_args()
    if args.tasks < 2 or args.seed < 0:
        parser.error("tasks must be >= 2 and seed must be nonnegative")
    if args.smoke and args.output == Path("results"):
        args.output = Path("results-smoke")
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep Matplotlib's cache inside the writable workspace, and use no GUI.
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from figures import plot_experiment, plot_tracking, plot_task_variability

    started = perf_counter()
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": args.seed, "smoke": args.smoke,
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "rng": "NumPy default_rng / PCG64; derived uint64 seeds via SeedSequence",
        "step_convention": "zero-based; permutation before action at t=5000",
        "trace_selection": "task index 0, fixed before inspecting results",
        "ci": "pointwise mean +/- 1.96 * across-task sample SD / sqrt(tasks)",
        "experiments": {},
    }
    all_data = {}
    for experiment_id, kind in enumerate(("stationary", "random_walk", "sudden_change")):
        stationary = kind == "stationary"
        methods = STATIONARY_METHODS if stationary else TRACKING_METHODS
        config = dict(kind=kind, tasks=32 if args.smoke else args.tasks, arms=10,
                      steps=(100 if stationary else 200) if args.smoke else
                      (1000 if stationary else 10000),
                      bin_width=(5 if args.smoke else (10 if stationary else 100)),
                      drift_std=0.01, change_step=100 if args.smoke else 5000,
                      initial_estimate=0.0, reward_std=1.0)
        seeds = make_seeds(args.seed, experiment_id, methods)
        print(f"Running {kind}: {config['tasks']:,} tasks, {config['steps']:,} steps",
              flush=True)
        experiment_start = perf_counter()
        data = simulate(config, methods, seeds)
        simulation_seconds = perf_counter() - experiment_start
        # Brief checks on actual results, not an extended test suite.
        assert np.isfinite(data["curve_mean"]).all()
        assert np.all(np.diff(data["curve_mean"][:, :, 2], axis=1) >= -1e-10)
        if kind == "random_walk":
            assert np.all(data["curve_mean"][:, 0, 1] == 1)
            assert np.all(data["curve_mean"][:, 0, 2] == 0)
        np.savez_compressed(args.output / f"{kind}.npz", **data)
        summary = summarize(data)
        (args.output / f"{kind}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        plot_experiment(data, config, args.output)
        if not stationary:
            plot_tracking(data, config, args.output)
        manifest["experiments"][kind] = {
            "config": config, "methods": [asdict(m) for m in methods], "seeds": seeds,
            "simulation_seconds": simulation_seconds,
            "simulation_save_and_plot_seconds": perf_counter() - experiment_start,
        }
        all_data[kind] = data
        print(f"  simulation: {simulation_seconds:.2f} s; final-window reward:", flush=True)
        final_window = "last_100" if stationary else "last_1000"
        for method, metrics in summary["windows"][final_window].items():
            print(f"    {method:16s} {metrics['reward']['mean']:.4f}", flush=True)
    plot_task_variability(all_data, args.output)
    manifest["total_seconds"] = perf_counter() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    source_dir = Path(__file__).resolve().parent
    manifest["source_sha256"] = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in ("bandits.py", "run_experiments.py", "figures.py")
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved results and figures to {args.output}; total {manifest['total_seconds']:.2f} s")


if __name__ == "__main__":
    main()
