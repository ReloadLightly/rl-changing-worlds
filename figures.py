"""Plot saved task-level statistics; no smoothing of standard errors."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


COLORS = ["#3f4a59", "#0072B2", "#009E73", "#D55E00"]
TITLES = {"stationary": "Stationary ten-armed testbed",
          "random_walk": "Changing values: independent random walks",
          "sudden_change": "Our extension: an unannounced permutation"}


def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.18, "figure.dpi": 110,
                         "savefig.dpi": 160})


def plot_experiment(data, config, output):
    style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for metric, ax in enumerate(axes):
        x = data["bin_ends"] if metric == 2 else data["bin_centers"]
        for m, name in enumerate(data["method_names"]):
            mean = data["bin_mean"][m, :, metric]
            half_width = 1.96 * data["bin_sem"][m, :, metric]
            ax.plot(x, mean, color=COLORS[m], label=name, linewidth=1.6)
            ax.fill_between(x, mean - half_width, mean + half_width,
                            color=COLORS[m], alpha=0.16, linewidth=0)
        if config["kind"] == "sudden_change":
            ax.axvline(config["change_step"], color="#8e44ad", linestyle="--", linewidth=1)
        ax.set_xlim(0, config["steps"] - 1)
    axes[0].set_ylabel("Mean reward")
    axes[1].set_ylabel("Optimal-action frequency")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylim(0, 1)
    axes[2].set_ylabel("Cumulative dynamic\npseudo-regret")
    axes[2].set_ylim(bottom=0)
    axes[2].set_xlabel("Decision step (zero-based)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="upper center",
               bbox_to_anchor=(0.5, 0.957), fontsize=9)
    fig.suptitle(TITLES[config["kind"]], fontsize=15, y=0.99)
    if config["kind"] == "random_walk" and config.get("initialization") in (None, "zeros"):
        axes[1].text(0.02, 0.95, "At step 0 all actions tie: 100% optimal (raw data).",
                     transform=axes[1].transAxes, va="top", fontsize=9)
    fig.text(0.5, 0.018,
             f"{config['tasks']:,} independent tasks · bands: pointwise 95% CIs for means\n"
             f"Reward/frequency: {config['bin_width']}-step task averages; regret: bin endpoints.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    fig.savefig(output / f"{config['kind']}.png")
    plt.close(fig)


def plot_tracking(data, config, output):
    """Fixed task 0, every action: no favorable trajectory is selected."""
    style()
    fig, axes = plt.subplots(5, 2, figsize=(13, 12), sharex=True, sharey=True)
    x = np.arange(config["steps"])
    average = list(data["method_names"]).index("sample average")
    constant = list(data["method_names"]).index("alpha=0.1")
    for arm, ax in enumerate(axes.flat):
        ax.plot(x, data["trace_q_true"][:, arm], color="black", lw=1.1, label="True value")
        ax.plot(x, data["trace_estimates"][average, :, arm], color=COLORS[0],
                lw=1, alpha=0.85, label="Sample-average estimate")
        ax.plot(x, data["trace_estimates"][constant, :, arm], color=COLORS[2],
                lw=0.9, alpha=0.85, label="alpha=0.1 estimate")
        if config["kind"] == "sudden_change":
            ax.axvline(config["change_step"], color="#8e44ad", ls="--", lw=1)
        ax.set_title(f"Action {arm}", loc="left", fontsize=10)
        ax.set_xlim(0, config["steps"] - 1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Action value")
    for ax in axes[-1]:
        ax.set_xlabel("Decision step (zero-based)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3)
    fig.suptitle(f"One fixed task, all ten actions — {TITLES[config['kind']]}", fontsize=14)
    fig.text(0.5, 0.015,
             "Task 0, selected in advance. Estimates are recorded before each action.\n"
             "Each learner sees only rewards from its own selected actions; flat estimates mean no update.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.935))
    fig.savefig(output / f"{config['kind']}_tracking.png")
    plt.close(fig)


def plot_task_variability(all_data, output):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (kind, data) in zip(axes, all_data.items()):
        w = list(data["window_names"]).index("all")
        regret = data["task_windows"][:, w, :, 2]
        # Show distribution spread separately from uncertainty in the mean.
        box = ax.boxplot(regret.T, tick_labels=data["method_names"],
                         whis=(10, 90), showfliers=False, patch_artist=True)
        for m, patch in enumerate(box["boxes"]):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.5)
        ax.scatter(np.arange(1, len(regret) + 1), regret.mean(axis=1),
                   marker="D", color="black", s=20, zorder=3)
        title = kind.replace("_", " ")
        ax.set_title(title + (" (our extension)" if kind == "sudden_change" else ""))
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Final cumulative dynamic pseudo-regret")
    fig.suptitle("Variation across independent tasks", fontsize=15)
    fig.legend([Line2D([], [], marker="D", color="black", linestyle="None")],
               ["Mean"], loc="upper right")
    fig.text(0.5, 0.018, "Boxes: 25th–75th percentiles; line: median; whiskers: 10th–90th percentiles.\n"
             "These show task-to-task spread, not confidence intervals. Panel scales differ.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(output / "task_variability.png")
    plt.close(fig)
