"""Section 2.8 plots: performance, exposure, and learned policies."""

import matplotlib.pyplot as plt
import numpy as np

from run_hidden_improvement import unvisited_curve


LABELS = ["G1: η=0.1, running baseline", "G2: η=0.4, running baseline",
          "G3: η=0.1, zero baseline", "G4: η=0.4, zero baseline",
          "E: ε=0.1, value α=0.1 (saved)", "U: UCB c=2, sample average (saved)"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#222222", "#8C6D00"]
STYLES = ["-", "-", "--", "--", "-.", ":"]


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)


def curve(ax, x, mean, half, method, log=False):
    ax.plot(x, mean, color=COLORS[method], ls=STYLES[method], lw=1.6,
            label=LABELS[method])
    lower = mean - half
    if log:
        lower = np.where(lower > 0, lower, np.nan)
    ax.fill_between(x, lower, mean + half, color=COLORS[method], alpha=0.10, linewidth=0)


def finish(fig, axes, output, filename, title, note, legend=True):
    fig.suptitle(title, fontsize=15)
    if legend:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.953),
                   ncol=2, fontsize=9, frameon=False)
    fig.text(0.5, 0.014, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.855 if legend else 0.94))
    fig.savefig(output / filename, dpi=160)
    plt.close(fig)


def plot_performance(data, references, output, stationary=False):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex="col")
    names = ("offset_0", "offset_4") if stationary else ("unchanged", "hidden_improvement")
    titles = ("Matched zero offset: N(0, 1)", "Figure 2.5 setup: N(4, 1)") if stationary else (
        "Unchanged world", "Hidden improvement at decision 5,000")
    for col, (name, title) in enumerate(zip(names, titles)):
        d = data[name]
        means, sems = d["bin_mean"], d["bin_sem"]
        if not stationary:
            means = np.concatenate((means, references[name]["bin_mean"]))
            sems = np.concatenate((sems, references[name]["bin_sem"]))
        for k in range(3):
            ax = axes[k, col]
            x = d["bin_ends"] if k == 2 else d["bin_centers"]
            scale = 100 if k == 1 else 1
            for m in range(len(means)):
                curve(ax, x, means[m, :, k] * scale, 1.96 * sems[m, :, k] * scale, m)
            if not stationary:
                ax.axvline(5000, color="black", ls=":", lw=1)
            if k == 1:
                ax.set_ylim(0, 100)
            tidy(ax)
        axes[0, col].set_title(title)
        axes[-1, col].set(xlabel="Decision (zero-based)", xlim=(0, 999 if stationary else 9999))
    for ax, label in zip(axes[:, 0], ["Mean reward", "Optimal action (%)", "Cumulative dynamic pseudo-regret"]):
        ax.set_ylabel(label)
    title = "Gradient bandits: reward offsets and the reward baseline" if stationary else (
        "Does direct policy learning find a hidden opportunity?")
    note = ("10-decision bins; pointwise 95% task intervals. Running-baseline policies overlap across offsets; rewards shift by four."
            if stationary else "100-decision bins; pointwise 95% task intervals. Saved E/U results use the same 2,000 worlds.\n"
            "Vertical line: intervention time (no intervention in the unchanged condition).")
    finish(fig, axes, output, "textbook_offsets.png" if stationary else "adaptation.png", title, note)


def plot_exposure(data, references, output):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    delays = np.concatenate((data["target_first_delay"], references["target_first_delay"]))
    for m, delay in enumerate(delays):
        mean, low, high = unvisited_curve(delay, 5000)
        x = np.arange(5001)
        ax.plot(x, mean, color=COLORS[m], ls=STYLES[m], lw=1.7, label=LABELS[m])
        ax.fill_between(x, low, high, color=COLORS[m], alpha=0.10, linewidth=0)
    ax.set(xlabel="Post-change decisions completed (k)", ylabel="Fraction still unvisited",
           xlim=(0, 5000), ylim=(0, 1.02))
    tidy(ax)
    finish(fig, np.array([ax]), output, "exposure.png", "First exposure to the improved arm",
           "Pointwise 95% Wilson intervals. A first visit at delay d changes this curve at k=d+1.\n"
           "Unvisited tasks are right-censored at 5,000 decisions. First exposure is not proof of learning.")


def plot_population_policy(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for col, name in enumerate(("unchanged", "hidden_improvement")):
        d = data[name]
        for k in range(2):
            ax = axes[k, col]
            for m in range(4):
                curve(ax, d["bin_centers"], d["policy_bin_mean"][m, :, k],
                      1.96 * d["policy_bin_sem"][m, :, k], m, log=k == 1)
            ax.axvline(5000, color="black", ls=":", lw=1)
            tidy(ax)
        axes[0, col].set(title=name.replace("_", " ").capitalize(), ylim=(0, np.log(10)))
        axes[1, col].set(yscale="log", xlabel="Decision (zero-based)", xlim=(0, 9999))
    axes[0, 0].set_ylabel("Mean policy entropy (nats)")
    axes[1, 0].set_ylabel("Mean probability of initially worst arm")
    finish(fig, axes, output, "policy_probabilities.png", "Policies can become confident without remaining adaptable",
           "100-decision task averages; pointwise 95% intervals. Entropy: 0 = one action, log(10) = uniform.\n"
           "Target identity is used only for evaluation. Population means can hide tasks with very small probabilities.")


def plot_task_policy(data, output):
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    target, favorite = int(data["target_arms"][0]), int(data["initial_q"][0].argmax())
    t = np.arange(data["trace_preferences"].shape[1])
    for m in range(4):
        for col, key in enumerate(("trace_probabilities", "trace_preferences")):
            ax = axes[m, col]
            for a, color, label in [(target, "#D55E00", f"Target (arm {target})"),
                                     (favorite, "#0072B2", f"Initially best (arm {favorite})")]:
                ax.plot(t, data[key][m, :, a], color=color, lw=1.1, label=label)
            ax.axvline(5000, color="black", ls=":", lw=1)
            ax.set_title(LABELS[m], loc="left", fontsize=10)
            ax.set_ylabel("Action probability (log scale)" if col == 0 else "Preference H")
            if col == 0:
                other = np.maximum(0, 1 - data[key][m, :, [target, favorite]].sum(axis=0))
                ax.plot(t, other, color="#777777", ls=":", lw=1,
                        label="Other arms (total probability)")
                ax.set_yscale("log")
            tidy(ax)
    for ax in axes[-1]:
        ax.set(xlabel="Decision (zero-based)", xlim=(0, 9999))
    finish(fig, axes, output, "task0_policy.png", "Hidden improvement: preferences and probabilities in fixed task 0",
           "Pre-decision values. Preferences are not reward estimates; other arms' preferences are omitted.\n"
           "Task 0 was fixed before examining outcomes. No uncertainty band: this is one illustrative task.")


def plot_all(data, references, output):
    plot_performance(data, references, output, stationary=True)
    plot_performance(data, references, output)
    plot_exposure(data["hidden_improvement"], references["hidden_improvement"], output)
    plot_population_policy(data, output)
    plot_task_policy(data["hidden_improvement"], output)
