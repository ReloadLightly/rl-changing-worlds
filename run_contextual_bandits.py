"""Run our original Section 2.9 experiment: informative versus unrelated cues."""

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

from bandits import mean_sem
from contextual_bandits import (METHODS, ContextGradientAgent, ContextValueAgent,
                                ContextWorld, context_seeds, evaluator_values)
from gradient_bandits import softmax
from run_experiments import describe
from run_exploration import write_json


CONFIG = dict(tasks=2000, arms=10, contexts=2, steps=5000, reward_std=1.0,
              bin_width=50, final_window=1000, initialization="independent N(0,1)",
              context_probability=0.5, trace_task=0, trace_window=[4000, 5000])
METRICS = ["reward", "full_information_regret", "cue_regret", "expected_full_information_regret"]
BENCHMARKS = ["full_information_optimum", "cue_optimum"]


def simulate(config, informative, seeds):
    tasks, arms, steps = (config[k] for k in ("tasks", "arms", "steps"))
    width, final = config["bin_width"], config["final_window"]
    rows = np.arange(tasks)
    world = ContextWorld(tasks, arms, seeds["environment"], config["reward_std"])
    agents = [(ContextGradientAgent if m.eta is not None else ContextValueAgent)(
        tasks, arms, m, seeds["agents"][m.name]) for m in METHODS]
    bins = (steps + width - 1) // width
    bin_mean, bin_sem = np.zeros((4, bins, 4)), np.zeros((4, bins, 4))
    bin_totals = np.zeros((4, tasks, 4))
    windows = np.array([[0, steps], [steps - final, steps]])
    task_windows = np.zeros((4, 2, tasks, 4))
    benchmark_windows = np.zeros((2, tasks, 2))
    benchmark_bins = np.zeros((tasks, 2))
    benchmark_mean, benchmark_sem = np.zeros((bins, 2)), np.zeros((bins, 2))
    actions_trace = np.zeros((4, steps), dtype=np.int16)
    rewards_trace = np.zeros((4, steps))
    contexts_trace, cues_trace = np.zeros(steps, dtype=np.int8), np.zeros(steps, dtype=np.int8)
    action_hashes = [hashlib.sha256() for _ in METHODS]
    context_hash, noise_hash = hashlib.sha256(), hashlib.sha256()

    for t in range(steps):
        contexts, cues, actual, potential_rewards, noise = world.draw(informative)
        conditional, full_best, cue_best, expected_full_best = evaluator_values(world.q, contexts, informative)
        benchmarks = np.column_stack((full_best, cue_best))
        benchmark_bins += benchmarks
        active_windows = [w for w, (start, stop) in enumerate(windows) if start <= t < stop]
        for w in active_windows:
            benchmark_windows[w] += benchmarks
        contexts_trace[t], cues_trace[t] = contexts[0], cues[0]
        context_hash.update(contexts.astype("u1").tobytes())
        noise_hash.update(noise.astype("<f8", copy=False).tobytes())
        bin_end = (t + 1) % width == 0 or t == steps - 1
        for m, agent in enumerate(agents):
            actions = agent.act(cues)
            rewards = potential_rewards[rows, actions]
            # Each regret subtracts values conditioned on the SAME information.
            values = np.column_stack((rewards, full_best - actual[rows, actions],
                                      cue_best - conditional[rows, actions],
                                      expected_full_best - conditional[rows, actions]))
            bin_totals[m] += values
            for w in active_windows:
                task_windows[m, w] += values
            if bin_end:
                bin_mean[m, t // width], bin_sem[m, t // width] = mean_sem(bin_totals[m] / (t % width + 1))
                bin_totals[m] = 0
            actions_trace[m, t], rewards_trace[m, t] = actions[0], rewards[0]
            action_hashes[m].update(actions.astype("u1").tobytes())
            agent.learn(actions, rewards)
        if bin_end:
            benchmark_mean[t // width], benchmark_sem[t // width] = mean_sem(benchmark_bins / (t % width + 1))
            benchmark_bins[:] = 0
        if (t + 1) % 1000 == 0:
            print(f"  {t+1:,}/{steps:,} decisions", flush=True)

    lengths = windows[:, 1] - windows[:, 0]
    task_windows /= lengths[None, :, None, None]
    benchmark_windows /= lengths[:, None, None]
    information_gap = (np.zeros(tasks) if informative else
                       world.q.max(axis=2).mean(axis=1) - world.q.mean(axis=1).max(axis=1))
    np.testing.assert_allclose(task_windows[..., 3], task_windows[..., 2] + information_gap,
                               rtol=1e-11, atol=1e-12)
    assert np.isfinite(task_windows).all() and np.all(task_windows[..., 1:] >= -1e-12)
    gradient = agents[-1]
    pi = softmax(gradient.preferences.reshape(-1, arms)).reshape(tasks, 2, arms)
    log_pi = np.zeros_like(pi)
    np.log(pi, out=log_pi, where=pi > 0)
    entropy = -(pi * log_pi).sum(axis=2)
    conditional_q = world.q if informative else np.repeat(world.q.mean(axis=1)[:, None, :], 2, axis=1)
    best_mask = conditional_q == conditional_q.max(axis=2, keepdims=True)
    optimal_probability = (pi * best_mask).sum(axis=2)
    starts = np.arange(0, steps, width)
    ends = np.minimum(starts + width, steps) - 1
    return dict(
        method_names=np.array([m.name for m in METHODS]), metric_names=np.array(METRICS),
        window_names=np.array(["all", "final_1000"]), window_bounds=windows, task_windows=task_windows,
        bin_mean=bin_mean, bin_sem=bin_sem, bin_centers=(starts + ends) / 2,
        benchmark_names=np.array(BENCHMARKS), task_benchmarks=benchmark_windows,
        benchmark_bin_mean=benchmark_mean, benchmark_bin_sem=benchmark_sem,
        task_information_gap=information_gap, q_true=world.q,
        trace_task=np.array(0), trace_contexts=contexts_trace, trace_cues=cues_trace,
        trace_actions=actions_trace, trace_rewards=rewards_trace,
        gradient_final_preferences=gradient.preferences, gradient_final_probabilities=pi,
        gradient_final_baselines=gradient.baselines, gradient_cue_counts=gradient.cue_counts,
        gradient_final_entropy=entropy, gradient_final_cue_optimal_probability=optimal_probability,
        action_sha256=np.array([h.hexdigest() for h in action_hashes]),
        context_sha256=np.array(context_hash.hexdigest()), noise_sha256=np.array(noise_hash.hexdigest()))


def summarize(data):
    result = {"windows": {}, "paired_method_differences": {},
              "information_gap": describe(data["task_information_gap"])}
    for w, window in enumerate(data["window_names"]):
        result["windows"][window] = {
            "methods": {name: {metric: describe(data["task_windows"][m, w, :, k])
                               for k, metric in enumerate(METRICS)}
                        for m, name in enumerate(data["method_names"])},
            "benchmarks": {name: describe(data["task_benchmarks"][w, :, k])
                           for k, name in enumerate(BENCHMARKS)}}
        result["paired_method_differences"][window] = {
            f"M{a+1}_minus_M{b+1}": {
                metric: describe(data["task_windows"][a, w, :, k] - data["task_windows"][b, w, :, k])
                for k, metric in enumerate(METRICS)}
            for a in range(1, 4) for b in range(a)}
    # Average the two cues within each task before computing uncertainty.
    result["gradient_final_policy"] = {
        "entropy_nats": describe(data["gradient_final_entropy"].mean(axis=1)),
        "cue_optimal_probability": describe(data["gradient_final_cue_optimal_probability"].mean(axis=1))}
    return result


def compare_conditions(informative, uninformative):
    for key in ("q_true", "context_sha256", "noise_sha256"):
        np.testing.assert_array_equal(informative[key], uninformative[key])
    # All-task action hashes and reward summaries, not just one illustrative trace.
    assert informative["action_sha256"][0] == uninformative["action_sha256"][0]
    np.testing.assert_array_equal(informative["task_windows"][0, :, :, :2], uninformative["task_windows"][0, :, :, :2])
    np.testing.assert_array_equal(informative["trace_actions"][0], uninformative["trace_actions"][0])
    result = {"world_draws_match": True, "blind_actions_and_rewards_identical": True,
              "paired_informative_minus_uninformative_reward": {}}
    for w, window in enumerate(informative["window_names"]):
        result["paired_informative_minus_uninformative_reward"][window] = {
            name: describe(informative["task_windows"][m, w, :, 0] - uninformative["task_windows"][m, w, :, 0])
            for m, name in enumerate(informative["method_names"])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/contextual_bandits"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from contextual_figures import plot_all
    if args.plot_only:
        data = {}
        for name in ("informative", "uninformative"):
            with np.load(args.output / f"{name}.npz", allow_pickle=False) as f:
                data[name] = dict(f)
        plot_all(data, args.output)
        return
    started = perf_counter()
    seeds = context_seeds(20260928)
    manifest = dict(
        design="Original experiment inspired by Sutton and Barto second edition, Section 2.9",
        started_utc=datetime.now(timezone.utc).isoformat(), experiment_id=9, master_seed=20260928,
        config=CONFIG, methods=[asdict(m) for m in METHODS], seeds=seeds,
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        platform=platform.platform(), primary="Mean reward over [4000,5000)",
        baseline_timing="Per observed cue, current reward included; denominator is that cue's count",
        intervals="Task mean +/- 1.96 SEM; paired task differences; exploratory, unadjusted",
        metrics="All saved window metrics are means per decision, including regrets",
        matching="Same task means, true contexts, potential noise and per-method RNG seeds; actions hashed across all tasks",
        hashes="Step-major task-order uint8 actions/contexts; little-endian float64 potential noise",
        trace_selection="Task 0, final 1000 decisions, fixed before seeing results",
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ("contextual_bandits.py", "run_contextual_bandits.py", "contextual_figures.py",
                                    "bandits.py", "gradient_bandits.py", "run_experiments.py", "run_exploration.py")},
        conditions={})
    completed = {}
    for name in ("informative", "uninformative"):
        print(f"Running {name}: 2,000 tasks × 5,000 decisions", flush=True)
        start = perf_counter()
        data = simulate(CONFIG, name == "informative", seeds)
        seconds = perf_counter() - start
        summary = summarize(data)
        np.savez_compressed(args.output / f"{name}.npz", **data)
        write_json(args.output / f"{name}_summary.json", summary)
        manifest["conditions"][name] = dict(informative=name == "informative", simulation_seconds=seconds)
        write_json(args.output / "manifest.json", manifest)
        completed[name] = data
        print(f"  {seconds:.2f} s; final-1,000 reward:", flush=True)
        for method, values in summary["windows"]["final_1000"]["methods"].items():
            print(f"    {method:40s} {values['reward']['mean']:.4f}", flush=True)
    write_json(args.output / "matched_comparison.json", compare_conditions(**completed))
    plot_all(completed, args.output)
    manifest["total_seconds"] = perf_counter() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "manifest.json", manifest)
    print(f"Saved {args.output}; total {manifest['total_seconds']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
