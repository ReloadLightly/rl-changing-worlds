"""Show learning, paired reward differences, information loss, and task-0 choices."""

import matplotlib.pyplot as plt
import numpy as np

from bandits import mean_sem


LABELS = ["M1: blind, α=0.1", "M2: cue values, α=0.1",
          "M3: cue values, sample averages", "M4: cue policy, η=0.1"]
COLORS = ["#666666", "#0072B2", "#009E73", "#D55E00"]
CONDITIONS = ["informative", "uninformative"]


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.17)


def save(fig, output, name):
    fig.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_learning(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey="row")
    for col, name in enumerate(CONDITIONS):
        d = data[name]
        for row, metric in enumerate((0, 2)):
            ax = axes[row, col]
            for m in range(4):
                mean, half = d["bin_mean"][m, :, metric], 1.96 * d["bin_sem"][m, :, metric]
                ax.plot(d["bin_centers"], mean, color=COLORS[m], label=LABELS[m])
                ax.fill_between(d["bin_centers"], mean-half, mean+half, color=COLORS[m], alpha=.12, linewidth=0)
            tidy(ax)
        for k, color, style in [(0, "#222222", "--"), (1, "#AA8800", ":")]:
            if name == "informative" and k == 1:
                continue  # The benchmarks coincide exactly here.
            mean, half = d["benchmark_bin_mean"][:, k], 1.96*d["benchmark_bin_sem"][:, k]
            label = "Full = cue optimum" if name == "informative" else ["Full-information optimum", "Cue optimum"][k]
            axes[0, col].plot(d["bin_centers"], mean, color=color, ls=style, lw=1.5, label=label)
            axes[0, col].fill_between(d["bin_centers"], mean-half, mean+half, color=color, alpha=.08, linewidth=0)
        axes[0, col].set_title(name.capitalize() + " cue")
        axes[0, col].legend(loc="lower right", fontsize=8, framealpha=.9)
        axes[1, col].set(xlabel="Decision (zero-based)", xlim=(0, 4999), ylim=(0, None))
    axes[0, 0].set_ylabel("Mean reward")
    axes[1, 0].set_ylabel("Cue-relative regret per decision")
    fig.suptitle("Recognizing the situation: learning with useful and unrelated cues", fontsize=15)
    fig.text(.5, .012, "50-decision task averages; pointwise 95% task intervals. Cue regret uses conditional expected action values.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    save(fig, output, "learning_curves.png")


def plot_pairs(data, output):
    pairs = [(1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, name in zip(axes, CONDITIONS):
        final = data[name]["task_windows"][:, 1, :, 0]
        for y, (a, b) in enumerate(pairs):
            mean, sem = mean_sem(final[a] - final[b])
            ax.errorbar(mean, y, xerr=1.96*sem, fmt="o", color=COLORS[a], capsize=4)
        ax.axvline(0, color="black", ls=":", lw=1)
        ax.set(title=name.capitalize() + " cue", xlabel="Paired final-1,000 reward difference",
               yticks=np.arange(len(pairs)), yticklabels=[f"M{a+1} − M{b+1}" for a, b in pairs])
        tidy(ax)
    axes[0].invert_yaxis()
    fig.suptitle("Does separating experience improve reward?", fontsize=15)
    fig.text(.5, .025, "M1: blind α=.1; M2: cue α=.1; M3: cue sample averages; M4: cue gradient η=.1.\n"
             "Positive favors the first method. Independent-task paired 95% intervals; shared horizontal scale.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .93))
    save(fig, output, "paired_comparisons.png")


def plot_information(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, name in zip(axes, CONDITIONS):
        d = data[name]
        gap = d["task_information_gap"]
        for m in range(4):
            cue_regret = d["task_windows"][m, 1, :, 2]
            mean, sem = mean_sem(gap + cue_regret)
            ax.bar(m, gap.mean(), color="#BBBBBB", label="Missing context information" if m == 0 else None)
            ax.bar(m, cue_regret.mean(), bottom=gap.mean(), color=COLORS[m])
            ax.errorbar(m, mean, yerr=1.96*sem, color="black", fmt="none", capsize=4)
        ax.set(title=name.capitalize() + " cue", xticks=range(4), xticklabels=["M1", "M2", "M3", "M4"])
        tidy(ax)
    axes[0].set_ylabel("Expected full-information regret per decision")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Missing information and failing to use available information", fontsize=15)
    fig.text(.5, .02, "Gray: unavoidable information gap. Color: final-1,000 cue-relative regret.\n"
             "Expectation over the true context given the cue; bars add exactly. Total intervals use paired task sums.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .93))
    save(fig, output, "information_gap.png")


def plot_task(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={"height_ratios": [1, 3]}, layout="constrained")
    q = data["informative"]["q_true"][0]
    vmax = abs(q).max()
    row_labels = [f"M{m+1}, cue {c}" for m in range(4) for c in range(2)]
    for col, name in enumerate(CONDITIONS):
        d = data[name]
        conditional = q if name == "informative" else np.repeat(q.mean(axis=0)[None, :], 2, axis=0)
        imq = axes[0, col].imshow(conditional, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        best = conditional.argmax(axis=1)
        axes[0, col].scatter(best, [0, 1], marker="*", c="black", s=100)
        axes[0, col].set(title=name.capitalize() + ": evaluator E[reward | cue, arm]",
                        yticks=[0, 1], yticklabels=["cue 0", "cue 1"], xticks=range(10))
        frequencies = np.zeros((8, 10))
        cues = d["trace_cues"][4000:5000]
        for m in range(4):
            actions = d["trace_actions"][m, 4000:5000]
            for c in range(2):
                frequencies[2*m+c] = np.bincount(actions[cues == c], minlength=10) / np.sum(cues == c)
        imf = axes[1, col].imshow(frequencies, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for row in range(8):
            axes[1, col].scatter(best[row % 2], row, marker="*", s=80, c="#D55E00", edgecolors="black", linewidths=.4)
        for boundary in [1.5, 3.5, 5.5]:
            axes[1, col].axhline(boundary, color="white", lw=2)
        axes[1, col].set(title="Task 0: action frequency within each observed cue",
                        xlabel="Arm", yticks=range(8), yticklabels=row_labels, xticks=range(10))
    fig.colorbar(imq, ax=list(axes[0]), shrink=.9, label="Conditional expected reward")
    fig.colorbar(imf, ax=list(axes[1]), shrink=.9, label="Selection frequency")
    fig.suptitle("Preselected task 0, decisions 4,000–4,999: association or irrelevant splitting?\n"
                 "Stars mark the cue-optimal arm. One illustrative task; no uncertainty interval.", fontsize=13)
    save(fig, output, "task0_by_cue.png")


def plot_all(data, output):
    plot_learning(data, output)
    plot_pairs(data, output)
    plot_information(data, output)
    plot_task(data, output)
