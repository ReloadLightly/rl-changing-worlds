"""Figures for the exploratory drift × observation-noise experiment."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

from figures import style


COLORS = ("#3f4a59", "#CC79A7", "#56B4E9", "#0072B2", "#009E73", "#E69F00", "#D55E00")


def plot_sweep(records, output):
    from run_memory_sweep import ALPHAS, CURVE_CONDITIONS, DRIFTS, METHODS, NOISES, condition_name

    style()
    by_condition = {(r["config"]["drift_std"], r["config"]["reward_std"]): r for r in records}
    means = np.empty((len(NOISES), len(DRIFTS), len(METHODS)))
    half_widths = np.empty_like(means)
    for n, noise in enumerate(NOISES):
        for d, drift in enumerate(DRIFTS):
            for m, method in enumerate(METHODS):
                stats = by_condition[drift, noise]["summary"]["methods"][method.name]
                primary = stats["metrics"]["final_regret_per_decision"]
                means[n, d, m] = primary["mean"]
                half_widths[n, d, m] = 1.96 * primary["sem"]
    excess = means - means.min(axis=2, keepdims=True)
    labels = ["Sample\naverage"] + [f"α={a:g}" for a in ALPHAS]
    for n, noise in enumerate(NOISES):
        fig, ax = plt.subplots(figsize=(12, 5.7))
        mesh = ax.imshow(excess[n], cmap="Blues", vmin=0, vmax=excess.max(), aspect="auto")
        for d, drift in enumerate(DRIFTS):
            summary = by_condition[drift, noise]["summary"]
            best = int(means[n, d].argmin())
            for m, method in enumerate(METHODS):
                close = m != best and method.name in summary["close_to_best_pointwise95"]
                label = f"{means[n, d, m]:.3f}{'†' if close else ''}\n±{half_widths[n, d, m]:.3f}"
                ax.text(m, d, label, ha="center", va="center", fontsize=9,
                        color="white" if excess[n, d, m] > 0.55 * excess.max() else "black")
            ax.add_patch(Rectangle((best - 0.48, d - 0.48), 0.96, 0.96,
                                   fill=False, edgecolor="#E69F00", linewidth=2.5))
        ax.set_xticks(range(len(METHODS)), labels)
        ax.set_yticks(range(len(DRIFTS)), [f"{d:g}" for d in DRIFTS])
        ax.set_xlabel("Learning method (all ε=0.1)")
        ax.set_ylabel("Drift standard deviation per decision")
        ax.grid(False)
        ax.set_title(f"Our memory sweep · reward noise σ={noise:g}\n"
                     "Cells: final-1,000 dynamic pseudo-regret per decision ± 95% CI half-width", pad=12)
        fig.colorbar(mesh, ax=ax, label="Excess over best mean in the SAME environment")
        fig.text(0.5, 0.025, "Orange outline: lowest measured mean. †: paired 95% interval against that method includes zero.\n"
                 "2,000 tasks per cell. Exploratory pointwise intervals; † does not establish equivalence. Color scales match across panels.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(output / f"heatmap_noise_{noise:g}.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for n, noise in enumerate(NOISES):
        best_all = means[n].argmin(axis=1)
        best_constant = means[n, :, 1:].argmin(axis=1)
        color = ("#0072B2", "#009E73", "#D55E00")[n]
        # Small offsets separate coincident methods without changing x labels.
        x = np.arange(len(DRIFTS)) + (n - 1) * 0.07
        axes[0].plot(x, best_all, marker=("o", "s", "^")[n], color=color,
                     label=f"Reward noise σ={noise:g}", alpha=0.85)
        axes[1].plot(x, np.array(ALPHAS)[best_constant], marker=("o", "s", "^")[n], color=color, alpha=0.85)
    axes[0].set_yticks(range(len(METHODS)), ["Sample average"] + [f"α={a:g}" for a in ALPHAS])
    axes[0].set_ylim(-0.3, len(METHODS) - 0.7)
    axes[0].set_title("Lowest mean among all seven learners")
    axes[0].set_ylabel("Best tested method (categorical axis)")
    axes[1].set_yscale("log")
    axes[1].set_yticks(ALPHAS, [f"{a:g}" for a in ALPHAS])
    axes[1].minorticks_off()
    axes[1].set_ylabel("Best tested constant α (log scale)")
    axes[1].set_title("Lowest mean among constant-rate learners")
    for ax in axes:
        ax.set_xticks(range(len(DRIFTS)), [f"{d:g}" for d in DRIFTS])
        ax.set_xlabel("Drift standard deviation (tested levels)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    fig.text(0.5, 0.025, "Selection uses final-1,000 mean pseudo-regret per decision. Lines connect tested conditions only.\n"
             "These are exploratory point-estimate winners, not universally optimal rates; inspect paired comparisons for close rankings.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    fig.savefig(output / "best_tested_rates.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex="col")
    for col, (drift, noise) in enumerate(CURVE_CONDITIONS):
        with np.load(output / f"{condition_name(drift, noise)}.npz", allow_pickle=False) as data:
            for metric in range(3):
                ax = axes[metric, col]
                x = data["bin_ends"] if metric == 2 else data["bin_centers"]
                for m, method in enumerate(METHODS):
                    mean = data["bin_mean"][m, :, metric]
                    hw = 1.96 * data["bin_sem"][m, :, metric]
                    ax.plot(x, mean, color=COLORS[m], linewidth=1.3, label=method.name)
                    ax.fill_between(x, mean - hw, mean + hw, color=COLORS[m], alpha=0.10, linewidth=0)
                ax.set_xlim(0, 9999)
            axes[0, col].set_title(f"Drift σ={drift:g} · reward noise σ={noise:g}")
            axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))
            axes[1, col].set_ylim(0, 1)
            axes[2, col].set_ylim(bottom=0)
            axes[2, col].set_xlabel("Decision step (zero-based)")
    axes[0, 0].set_ylabel("Mean reward")
    axes[1, 0].set_ylabel("Optimal-action frequency")
    axes[2, 0].set_ylabel("Cumulative dynamic\npseudo-regret")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Our memory sweep: three conditions selected before running", fontsize=15)
    fig.text(0.5, 0.015, "All conditions start from the same N(0,1) values. 2,000 tasks; bands are pointwise 95% CIs.\n"
             "Reward/frequency: 100-step task averages; regret: cumulative at bin endpoints. Panel scales differ.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.89))
    fig.savefig(output / "learning_curves.png")
    plt.close(fig)
