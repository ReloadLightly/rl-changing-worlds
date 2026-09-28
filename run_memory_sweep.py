"""Our Chapter 2 extension: cross environmental drift with reward noise.

Run: python run_memory_sweep.py
Resume completed conditions: python run_memory_sweep.py --resume
Rebuild tables/figures from saved conditions: python run_memory_sweep.py --plot-only
"""

import argparse
import csv
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


DRIFTS = (0, 0.001, 0.003, 0.01, 0.03)
NOISES = (0.5, 1.0, 2.0)
ALPHAS = (0.003, 0.01, 0.03, 0.1, 0.3, 0.5)
METHODS = [Method("sample average", 0.1)] + [
    Method(f"alpha={alpha:g}", 0.1, alpha) for alpha in ALPHAS
]
# Selected before running or examining any sweep result.
CURVE_CONDITIONS = ((0, 1.0), (0.01, 1.0), (0.01, 2.0))
METRICS = ("final_regret_per_decision", "total_regret", "final_reward", "final_optimal_fraction")


def condition_name(drift, noise):
    return f"drift_{drift:g}_noise_{noise:g}"


def task_outcomes(data):
    """Return method × independent task × metric, in METRICS order."""
    w = list(data["window_names"]).index("last_1000")
    start, stop = data["window_bounds"][w]
    final = data["task_windows"][:, w]
    full = data["task_windows"][:, list(data["window_names"]).index("all")]
    return np.stack((final[:, :, 2] / (stop - start), full[:, :, 2],
                     final[:, :, 0], final[:, :, 1]), axis=-1)


def summarize(outcomes):
    primary = outcomes[:, :, 0]
    best = int(primary.mean(axis=1).argmin())
    best_constant = int(primary[1:].mean(axis=1).argmin()) + 1
    ordered = np.argsort(primary.mean(axis=1))
    runner_up = int(ordered[1])
    constant_runner_up = int(np.argsort(primary[1:].mean(axis=1))[1]) + 1
    return {
        "methods": {
            method.name: {
                "metrics": {metric: describe(outcomes[m, :, k]) for k, metric in enumerate(METRICS)},
                "paired_minus_sample_average": {
                    metric: describe(outcomes[m, :, k] - outcomes[0, :, k])
                    for k, metric in enumerate(METRICS)
                },
                "paired_primary_minus_best": describe(primary[m] - primary[best]),
            } for m, method in enumerate(METHODS)
        },
        "best_tested_method": METHODS[best].name,
        "best_tested_constant_alpha": METHODS[best_constant].alpha,
        "constant_runner_up": METHODS[constant_runner_up].name,
        "constant_runner_up_minus_best_constant": describe(primary[constant_runner_up] - primary[best_constant]),
        "runner_up": METHODS[runner_up].name,
        "runner_up_minus_best": describe(primary[runner_up] - primary[best]),
        "close_to_best_pointwise95": [METHODS[m].name for m in range(len(METHODS))
                                      if describe(primary[m] - primary[best])["ci95"][0] <= 0],
    }


def write_tables(records, output):
    """One row per condition/method, including all four outcomes and paired CIs."""
    rows = []
    for record in records:
        for method, stats in record["summary"]["methods"].items():
            row = {"drift_std": record["config"]["drift_std"],
                   "reward_std": record["config"]["reward_std"], "method": method}
            for metric in METRICS:
                for suffix, values in (("", stats["metrics"][metric]),
                                       ("_minus_sample_average", stats["paired_minus_sample_average"][metric])):
                    key = metric + suffix
                    row.update({key: values["mean"], key + "_ci_low": values["ci95"][0],
                                key + "_ci_high": values["ci95"][1]})
            rows.append(row)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path, default=Path("results/memory_sweep"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from memory_sweep_figures import plot_sweep

    started = perf_counter()
    sources = ("bandits.py", "run_memory_sweep.py", "run_experiments.py")
    source_dir = Path(__file__).resolve().parent
    source_hashes = {p: hashlib.sha256((source_dir / p).read_bytes()).hexdigest() for p in sources}
    # The SAME seed dictionary for all 15 conditions pairs the underlying draws.
    seeds = make_seeds(args.seed, 3, METHODS)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    master_seed=args.seed, experiment_id=3, seeds=seeds,
                    drifts=DRIFTS, reward_stds=NOISES, methods=[asdict(m) for m in METHODS],
                    curve_conditions=CURVE_CONDITIONS, source_sha256=source_hashes,
                    python=platform.python_version(), numpy=np.__version__,
                    matplotlib=matplotlib.__version__, platform=platform.platform(),
                    rng="NumPy default_rng / PCG64; SeedSequence-derived uint64 seeds",
                    pairing="Same environment and per-method agent streams restart in every condition; environment and agents use separate streams",
                    primary_outcome="Mean dynamic pseudo-regret per decision, steps 9000:10000",
                    uncertainty="Across-task mean +/- 1.96 SE; exploratory pointwise paired comparisons, no selection or multiplicity adjustment",
                    conditions=[])
    records = []
    for noise in NOISES:
        for drift in DRIFTS:
            name = condition_name(drift, noise)
            config = dict(kind="random_walk", tasks=2000, steps=10000, arms=10,
                          initialization="normal", initial_estimate=0.0,
                          drift_std=drift, reward_std=noise, change_step=5000, bin_width=100)
            record_path = args.output / f"{name}.json"
            data_path = args.output / f"{name}.npz"
            identity = dict(config=config, seeds=seeds, methods=manifest["methods"],
                            source_sha256=source_hashes, numpy=np.__version__)
            if record_path.exists() and (args.resume or args.plot_only):
                record = json.loads(record_path.read_text())
                if any(record.get(key) != value for key, value in identity.items()) or not data_path.exists():
                    raise ValueError(f"Saved condition does not match current run: {name}; use a new output directory")
                print(f"Using completed {name}", flush=True)
            else:
                if args.plot_only:
                    raise FileNotFoundError(f"Missing completed condition: {name}")
                if record_path.exists() or data_path.exists():
                    raise FileExistsError(f"{name} already exists; use --resume or a new output directory")
                print(f"Running {name}: 2,000 tasks × 10,000 steps × 7 learners", flush=True)
                condition_started = datetime.now(timezone.utc).isoformat()
                condition_start = perf_counter()
                data = simulate(config, METHODS, seeds)
                seconds = perf_counter() - condition_start
                outcomes = task_outcomes(data)
                assert np.isfinite(outcomes).all() and (outcomes[:, :, :2] >= 0).all()
                summary = summarize(outcomes)
                # Keep compact task outcomes and bin statistics, not all trajectories.
                compact = {key: data[key] for key in (
                    "method_names", "bin_mean", "bin_sem", "bin_centers", "bin_ends")}
                compact.update(task_outcomes=outcomes, outcome_names=np.array(METRICS))
                np.savez_compressed(data_path, **compact)
                record = dict(**identity, started_utc=condition_started,
                    finished_utc=datetime.now(timezone.utc).isoformat(), simulation_seconds=seconds,
                    simulation_and_save_seconds=perf_counter() - condition_start, summary=summary)
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                print(f"Saved {name}: {seconds:.1f} s; best tested = {summary['best_tested_method']}", flush=True)
            records.append(record)
            manifest["conditions"].append(name)
            manifest["completed_conditions"] = len(records)
            manifest["simulation_seconds"] = sum(r["simulation_seconds"] for r in records)
            manifest["condition_pipeline_seconds"] = sum(r["simulation_and_save_seconds"] for r in records)
            # A usable manifest and both condition files exist after each completion.
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    write_tables(records, args.output)
    plot_start = perf_counter()
    plot_sweep(records, args.output)
    manifest.update(plot_seconds=perf_counter() - plot_start,
                    invocation_seconds=perf_counter() - started,
                    invocation_mode="plot-only" if args.plot_only else ("resume" if args.resume else "full"),
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    plot_source_sha256=hashlib.sha256((source_dir / "memory_sweep_figures.py").read_bytes()).hexdigest())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed {len(records)} conditions; simulation {manifest['simulation_seconds']:.1f} s; "
          f"this invocation {manifest['invocation_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
