"""Plots for the six fixed exploration settings; intervals use independent tasks."""

import matplotlib.pyplot as plt
import numpy as np


LABELS = ["1: ε=0.1, sample average", "2: ε=0.1, α=0.1",
          "3: greedy, α=0.1, Q₀=0", "4: optimistic greedy, α=0.1, Q₀=5",
          "5: UCB, sample average", "6: UCB, α=0.1 (our variant)"]
COLORS = ["#0072B2", "#D55E00", "#777777", "#009E73", "#CC79A7", "#6A3D9A"]
STYLES = ["-", "-", "--", "-", "--", "-"]


def line(ax, x, mean, sem, method):
    ax.plot(x, mean, color=COLORS[method], ls=STYLES[method], lw=1.6,
            label=LABELS[method])
    ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem,
                    color=COLORS[method], alpha=0.12, linewidth=0)


def tidy(ax):
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def plot_curves(data, config, output):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for k, (ax, label) in enumerate(zip(axes, ["Mean reward", "Optimal action (%)",
                                             "Cumulative dynamic pseudo-regret"])):
        x = data["bin_ends"] if k == 2 else data["bin_centers"]
        scale = 100 if k == 1 else 1
        for m in range(6):
            line(ax, x, data["bin_mean"][m, :, k] * scale,
                 data["bin_sem"][m, :, k] * scale, m)
        ax.set_ylabel(label)
        tidy(ax)
        if config["kind"] == "sudden_change":
            ax.axvline(config["change_step"], color="black", ls=":", lw=1.2)
    axes[-1].set_xlabel("Decision (zero-based)")
    axes[-1].set_xlim(0, config["steps"] - 1)
    title = {"stationary": "A. Stationary", "random_walk": "B. Gradual change",
             "sudden_change": "C. Sudden change — our extension"}[config["kind"]]
    fig.suptitle(f"{title}: exploration and value estimation", fontsize=16, y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.962),
               ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.012,
             f"{config['tasks']:,} independent tasks; pointwise 95% intervals; "
             f"{config['bin_width']}-decision bins. UCB: c=2, lifetime counts.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    fig.savefig(output / f"{config['kind']}.png", dpi=160)
    plt.close(fig)


def plot_textbook_pairs(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, (methods, metric, title, ylabel) in enumerate([
        ([3, 1], 1, "Figure 2.3 comparison: optimism", "Optimal action (%)"),
        ([4, 0], 0, "Figure 2.4 comparison: UCB", "Mean reward"),
    ]):
        for row, stop in enumerate([1000, 100]):
            ax = axes[row, col]
            scale = 100 if metric == 1 else 1
            for m in methods:
                line(ax, np.arange(stop), data["curve_mean"][m, :stop, metric] * scale,
                     data["curve_sem"][m, :stop, metric] * scale, m)
            ax.set(xlabel="Decision (zero-based)", ylabel=ylabel, xlim=(0, stop - 1))
            ax.set_title(title if row == 0 else "First 100 decisions (same run)")
            if row == 0:
                ax.legend(frameon=False, fontsize=9, loc="lower right")
            tidy(ax)
    fig.suptitle("Chapter 2 stationary comparisons at the textbook settings", fontsize=15)
    fig.text(0.5, 0.012,
             "Unbinned means and pointwise 95% intervals; shared worlds, independent "
             "learner streams. These are new simulations.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output / "stationary_textbook_pairs.png", dpi=160)
    plt.close(fig)


def plot_action_raster(data, window, output):
    start, stop = window
    x = np.arange(start, stop)
    best = data["trace_q_true"][start:stop].argmax(axis=1)
    fig, axes = plt.subplots(6, 1, figsize=(13, 10), sharex=True, sharey=True)
    for m, ax in enumerate(axes):
        ax.scatter(x, data["trace_actions"][m, start:stop], s=4, color=COLORS[m],
                   alpha=0.65, rasterized=True)
        ax.step(x, best, where="post", color="black", lw=1.1, ls="--")
        ax.axvline(5000, color="black", lw=1.1, ls=":")
        ax.set_title(LABELS[m], loc="left", fontsize=10, pad=3)
        ax.set(ylim=(-0.5, 9.5), yticks=[0, 3, 6, 9], ylabel="Arm")
        tidy(ax)
    axes[-1].set(xlabel="Decision (zero-based)", xlim=(start, stop - 1))
    fig.suptitle("Task 0: actions around the unannounced permutation", fontsize=15)
    fig.text(0.5, 0.013, "Dots: selected arms. Dashed line: true best arm "
             "(evaluator only). Vertical line: change at 5,000.\n"
             "Task 0 and the window [4,500, 6,500) were fixed before inspecting results; "
             "one illustrative world, not a population average.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(output / "sudden_action_raster.png", dpi=160)
    plt.close(fig)
