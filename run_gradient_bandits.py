"""Section 2.8: reward offsets, policy confidence, and hidden opportunities."""

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

from bandits import make_seeds, simulate
from gradient_bandits import METHODS
from run_experiments import describe
from run_exploration import write_json
from run_hidden_improvement import compact as compact_hidden, median_delay, wilson_interval


SAVED_DIR = Path(__file__).with_name("results") / "hidden_improvement"
REFERENCE_NAMES = ["alpha=0.1", "UCB sample average"]
MASTER_SEED = 20260928


def expected_config(kind, stationary=False, offset=0.0):
    return dict(kind=kind, tasks=2000, arms=10, steps=1000 if stationary else 10000,
                bin_width=10 if stationary else 100, drift_std=0.0, change_step=5000,
                initialization="normal", reward_std=1.0, improvement_gap=0.5,
                reward_offset=offset,
                policy_snapshot_steps=[0, 999] if stationary else [0, 4999, 5000, 5999, 9999])


def load_references():
    """Refuse unpaired comparisons: verify recorded design, method settings, and streams."""
    manifest = json.loads((SAVED_DIR / "manifest.json").read_text())
    assert manifest["experiment_id"] == 7 and manifest["master_seed"] == MASTER_SEED
    assert manifest["numpy"] == np.__version__, "Use the saved run's NumPy version for pairing"
    seeds = make_seeds(MASTER_SEED, 7, METHODS)
    assert seeds["environment"] == manifest["seeds"]["environment"]
    from bandits import Method
    references = [Method("alpha=0.1", 0.1, 0.1),
                  Method("UCB sample average", 0.0, ucb_c=2.0)]
    by_name = {m["name"]: m for m in manifest["methods"]}
    old_seeds = make_seeds(MASTER_SEED, 7, references)
    for method in references:
        assert by_name[method.name] == asdict(method)
        assert old_seeds["agents"][method.name] == manifest["seeds"]["agents"][method.name]
    saved = {}
    for kind in ("unchanged", "hidden_improvement"):
        expected = expected_config(kind)
        recorded = manifest["conditions"][kind]["config"]
        assert recorded == {k: v for k, v in expected.items()
                            if k not in ("reward_offset", "policy_snapshot_steps")}
        with np.load(SAVED_DIR / f"{kind}.npz", allow_pickle=False) as archive:
            data = dict(archive)
        indices = [list(data["method_names"]).index(name) for name in REFERENCE_NAMES]
        for key in ("task_outcomes", "bin_mean", "bin_sem", "target_first_delay"):
            data[key] = data[key][indices]
        data["method_names"] = np.array(REFERENCE_NAMES)
        saved[kind] = data
    return saved


def outcome_summary(names, values):
    result = {}
    for k, name in enumerate(names):
        v = values[:, k]
        if name.startswith(("visited_", "unvisited_")):
            low, high = wilson_interval(v.sum(), len(v))
            result[name] = {"mean": float(v.mean()), "ci95": [float(low), float(high)]}
        else:
            result[name] = describe(v)
    return result


def summarize(data, reference=None):
    names, values = data["outcome_names"], data["task_outcomes"]
    result = {"methods": {}, "paired_gradient_comparisons": {}, "policy_snapshots": {}}
    for m, name in enumerate(data["method_names"]):
        result["methods"][name] = outcome_summary(names, values[m])
    # Zero minus running baseline at each eta, and eta .4 minus .1 with baseline.
    for a, b in [(2, 0), (3, 1), (1, 0)]:
        result["paired_gradient_comparisons"][f"G{a+1}_minus_G{b+1}"] = {
            name: describe(values[a, :, k] - values[b, :, k]) for k, name in enumerate(names)}
    for s, step in enumerate(data["policy_snapshot_steps"]):
        result["policy_snapshots"][str(step)] = {
            method.name: {metric: describe(data["task_policy_snapshots"][m, s, :, k])
                          for k, metric in enumerate(data["policy_metric_names"])}
            for m, method in enumerate(METHODS)}
    if reference is not None:
        np.testing.assert_array_equal(names, reference["outcome_names"])
        for key in ("initial_q", "target_arms", "trace_q_true", "bin_centers", "bin_ends"):
            np.testing.assert_array_equal(data[key], reference[key])
        result["saved_references"] = {
            name: outcome_summary(names, reference["task_outcomes"][i])
            for i, name in enumerate(REFERENCE_NAMES)}
        result["paired_vs_saved_references"] = {
            method.name: {
                name: {metric: describe(values[m, :, k] - reference["task_outcomes"][i, :, k])
                       for k, metric in enumerate(names)}
                for i, name in enumerate(REFERENCE_NAMES)}
            for m, method in enumerate(METHODS)}
        result["median_first_visit_delays"] = {
            method.name: median_delay(data["target_first_delay"][m], 5000)
            for m, method in enumerate(METHODS)}
    return result


def compact_stationary(data):
    windows = {name: data["task_windows"][:, w]
               for w, name in enumerate(data["window_names"])}
    names = ["mean_reward", "optimal_fraction", "total_regret",
             "last_100_reward", "last_100_optimal_fraction", "last_100_regret_per_decision"]
    values = np.concatenate((windows["all"], windows["last_100"]), axis=2)
    values[:, :, -1] /= 100
    omit = {"task_windows", "window_names", "window_bounds"}
    result = {key: value for key, value in data.items() if key not in omit}
    result.update(outcome_names=np.array(names), task_outcomes=values)
    return result


def compare_offsets(zero, shifted):
    result = {}
    for m, method in enumerate(METHODS):
        difference = shifted["task_outcomes"][m] - zero["task_outcomes"][m]
        # Center reward differences by the known +4 change in measurement units.
        difference[:, [0, 3]] -= 4
        result[method.name] = {
            "all_task_action_sequences_identical": bool(zero["action_sha256"][m] == shifted["action_sha256"][m]),
            "paired_offset_effects_reward_centered_by_four": {
                name: describe(difference[:, k]) for k, name in enumerate(zero["outcome_names"])}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/gradient_bandits"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from gradient_figures import plot_all
    references = load_references()
    conditions = [("offset_0", expected_config("stationary", True, 0), 8),
                  ("offset_4", expected_config("stationary", True, 4), 8),
                  ("unchanged", expected_config("unchanged"), 7),
                  ("hidden_improvement", expected_config("hidden_improvement"), 7)]
    if args.plot_only:
        data = {}
        for name, _, _ in conditions:
            with np.load(args.output / f"{name}.npz", allow_pickle=False) as archive:
                data[name] = dict(archive)
        plot_all(data, references, args.output)
        return
    started = perf_counter()
    manifest = dict(
        started_utc=datetime.now(timezone.utc).isoformat(), master_seed=MASTER_SEED,
        methods=[asdict(m) for m in METHODS], python=platform.python_version(),
        numpy=np.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(),
        baseline_timing="Running average includes current reward, before preference update",
        policy_timing="All recorded H, probabilities, entropy are before the decision; recorded baseline includes its reward",
        intervals="Task mean +/- 1.96 SEM; Wilson visit proportions; pointwise, unadjusted",
        censoring="delay -1 means not visited within 5000 post-change decisions; null median means not reached",
        trace_selection="task 0 fixed before examining results",
        action_hash="SHA256 of step-major, task-order actions as little-endian int16; no action cube retained",
        saved_reference_files_sha256={name: hashlib.sha256((SAVED_DIR / name).read_bytes()).hexdigest()
                                      for name in ("manifest.json", "unchanged.npz", "hidden_improvement.npz")},
        saved_reference_methods=REFERENCE_NAMES,
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ("bandits.py", "gradient_bandits.py", "run_gradient_bandits.py",
                                    "gradient_figures.py", "run_experiments.py", "run_exploration.py",
                                    "run_hidden_improvement.py")}, conditions={})
    completed = {}
    for name, config, experiment_id in conditions:
        seeds = make_seeds(MASTER_SEED, experiment_id, METHODS)
        print(f"Running {name}: 2,000 tasks × {config['steps']:,} decisions", flush=True)
        start = perf_counter()
        full = simulate(config, METHODS, seeds, gradient=True)
        seconds = perf_counter() - start
        assert np.isfinite(full["curve_mean"]).all()
        assert np.isfinite(full["trace_preferences"]).all()
        assert np.all(np.diff(full["curve_mean"][:, :, 2], axis=1) >= -1e-10)
        data = compact_stationary(full) if experiment_id == 8 else compact_hidden(full)
        del full
        if name == "offset_4":
            np.testing.assert_allclose(data["initial_q"] - 4, completed["offset_0"]["initial_q"], atol=1e-15)
            write_json(args.output / "offset_comparison.json", compare_offsets(completed["offset_0"], data))
        if name == "hidden_improvement":
            for key in ("target_first_delay", "target_pre_visits"):
                np.testing.assert_array_equal(data[key], completed["unchanged"][key])
            np.testing.assert_array_equal(data["bin_mean"][:, :50], completed["unchanged"]["bin_mean"][:, :50])
        summary = summarize(data, references.get(name))
        np.savez_compressed(args.output / f"{name}.npz", **data)
        write_json(args.output / f"{name}_summary.json", summary)
        manifest["conditions"][name] = dict(config=config, experiment_id=experiment_id,
                                            seeds=seeds, simulation_seconds=seconds)
        write_json(args.output / "manifest.json", manifest)
        completed[name] = data
        metric = "total_regret" if experiment_id == 8 else "post_regret_per_decision"
        print(f"  {seconds:.2f} s; {metric}:", flush=True)
        for method, metrics in summary["methods"].items():
            print(f"    {method:35s} {metrics[metric]['mean']:.4f}", flush=True)
    plot_all(completed, references, args.output)
    manifest["total_seconds"] = perf_counter() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "manifest.json", manifest)
    print(f"Saved to {args.output}; total {manifest['total_seconds']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
