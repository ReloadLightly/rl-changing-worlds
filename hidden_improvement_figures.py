"""Learning, first exposure, and the reward cost/benefit of persistent exploration."""

import matplotlib.pyplot as plt
import numpy as np

from bandits import mean_sem
from exploration_figures import LABELS as OLD_LABELS, COLORS as OLD_COLORS, tidy
from run_hidden_improvement import unvisited_curve


LABELS = OLD_LABELS + ["7: optimistic, ε=0.01, α=0.1", "8: optimistic, ε=0.1, α=0.1"]
COLORS = OLD_COLORS + ["#E69F00", "#222222"]
STYLES = ["-", "-", "--", "-", "--", "-", "--", "-."]


def curve(ax, x, mean, low, high, m):
    ax.plot(x, mean, color=COLORS[m], ls=STYLES[m], lw=1.6, label=LABELS[m])
    ax.fill_between(x, low, high, color=COLORS[m], alpha=0.10, linewidth=0)


def plot_curves(data, kind, output):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for k, ax in enumerate(axes):
        x = data["bin_ends"] if k == 2 else data["bin_centers"]
        scale = 100 if k == 1 else 1
        for m in range(8):
            mean = data["bin_mean"][m, :, k] * scale
            half = 1.96 * data["bin_sem"][m, :, k] * scale
            curve(ax, x, mean, mean - half, mean + half, m)
        ax.axvline(5000, color="black", ls=":", lw=1)
        tidy(ax)
    for ax, label in zip(axes, ["Mean reward", "Optimal action (%)", "Cumulative dynamic pseudo-regret"]):
        ax.set_ylabel(label)
    axes[-1].set(xlabel="Decision (zero-based)", xlim=(0, 9999))
    title = "A. Unchanged world" if kind == "unchanged" else "B. Hidden improvement"
    fig.suptitle(title + ": will the agent explore when nothing goes wrong?", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=3, frameon=False, fontsize=9)
    fig.text(0.5, 0.012, "100-decision bins; task-level pointwise 95% intervals. "
             "Vertical line marks the intervention time (no intervention in A).", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    fig.savefig(output / f"{kind}.png", dpi=160)
    plt.close(fig)


def plot_unvisited(data, output):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for m, delays in enumerate(data["target_first_delay"]):
        mean, low, high = unvisited_curve(delays, 5000)
        curve(ax, np.arange(5001), mean, low, high, m)
    ax.set(xlabel="Post-change decisions completed (k)", ylabel="Fraction still unvisited",
           xlim=(0, 5000), ylim=(0, 1.02))
    fig.suptitle("First exposure to the initially worst arm", fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, fontsize=9,
               loc="upper center", bbox_to_anchor=(0.5, 0.94))
    tidy(ax)
    fig.text(0.5, 0.018, "Pointwise 95% Wilson intervals. A visit at delay d appears after d+1 decisions.\n"
             "First-visit delays match exactly across both conditions; unvisited tasks are censored at the horizon.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.78))
    fig.savefig(output / "still_unvisited.png", dpi=160)
    plt.close(fig)


def plot_cost_benefit(all_data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, kind, sign, title in zip(
            axes, ("unchanged", "hidden_improvement"), (-1, 1),
            ("Unchanged world: reward cost", "Hidden improvement: reward benefit")):
        data = all_data[kind]
        k = list(data["outcome_names"]).index("post_reward")
        for y, m in enumerate((6, 7)):
            difference = sign * (data["task_outcomes"][m, :, k] - data["task_outcomes"][3, :, k])
            mean, sem = mean_sem(difference)
            ax.errorbar(mean, y, xerr=1.96 * sem, fmt="o", color=COLORS[m], capsize=5)
        ax.axvline(0, color="gray", ls=":")
        ax.set(yticks=[0, 1], yticklabels=["7: ε=0.01", "8: ε=0.1"],
               ylim=(-0.6, 1.6), title=title, xlabel="Reward per decision, [5,000, 10,000)")
        tidy(ax)
    fig.suptitle("Adding persistent exploration to optimistic greedy (4)", fontsize=14)
    fig.text(0.5, 0.012, "Both settings keep Q₀=5 and α=0.1. Paired task differences with 95% intervals.\n"
             "Cost = reward(4) − reward(explorer); benefit = reward(explorer) − reward(4). Shared axis scale.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(output / "exploration_cost_benefit.png", dpi=160)
    plt.close(fig)


def plot_raster(data, output):
    start, stop = 4500, 10000  # Fixed before inspecting task 0.
    target = int(data["target_arms"][0])
    fig, axes = plt.subplots(8, 1, figsize=(13, 12), sharex=True, sharey=True)
    for m, ax in enumerate(axes):
        ax.scatter(np.arange(start, stop), data["trace_actions"][m, start:stop],
                   s=3, color=COLORS[m], alpha=0.65, rasterized=True)
        ax.axhline(target, color="#B22222", ls="--", lw=1.0)
        ax.axvline(5000, color="black", ls=":", lw=1)
        ax.set(ylim=(-0.5, 9.5), yticks=sorted({0, target, 9}), ylabel="Arm")
        ax.set_title(LABELS[m], loc="left", fontsize=9, pad=3)
        tidy(ax)
    axes[-1].set(xlabel="Decision (zero-based)", xlim=(start, stop - 1))
    fig.suptitle(f"Hidden improvement, fixed task 0: target arm {target}", fontsize=15)
    fig.text(0.5, 0.013, "Dots: selected actions. Red dashed line: target (initially worst; best after 5,000).\n"
             "Vertical line: unannounced improvement. Task and window were fixed before inspecting results.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output / "task0_action_raster.png", dpi=160)
    plt.close(fig)


def plot_all(all_data, output):
    for kind, data in all_data.items():
        plot_curves(data, kind, output)
    plot_unvisited(all_data["hidden_improvement"], output)
    plot_cost_benefit(all_data, output)
    plot_raster(all_data["hidden_improvement"], output)
