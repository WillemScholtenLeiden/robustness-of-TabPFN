"""
Plotting functions for Experiment 10: Adversarial Transferability.

Plots heatmaps of transfer rates (source model x target model) for each
attack type and epsilon, plus a summary line plot of transfer rate vs epsilon.

Usage:
    data = load_experiment_data("transferability")
    plot_experiment_10(data, save_dir="figures")
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Any
from helpers import load_experiment_data
from plot_style import PLOT_STYLE

COLORS = {
    "TabPFN": "#0072B2",
    "LogReg": "#D55E00",
    "SVM": "#009E73",
}

LINE_STYLES = {
    "TabPFN": "-o",
    "LogReg": "-s",
    "SVM": "-^",
}


def _apply_style():
    mpl.rcParams.update(PLOT_STYLE)


def _build_matrix(transfer_rates: dict, model_names: list[str]) -> np.ndarray:
    """Build a source x target transfer-rate matrix from a nested dict."""
    n = len(model_names)
    mat = np.full((n, n), np.nan)
    for i, src in enumerate(model_names):
        for j, tgt in enumerate(model_names):
            mat[i, j] = transfer_rates[src][tgt]
    return mat


def _plot_heatmap(
    mat: np.ndarray,
    model_names: list[str],
    ax: plt.Axes,
    title: str,
) -> None:
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd", aspect="equal")

    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Target model")
    ax.set_ylabel("Source model")
    ax.set_title(title)

    for i in range(len(model_names)):
        for j in range(len(model_names)):
            val = mat[i, j]
            if np.isnan(val):
                text = "n/a"
            else:
                text = f"{val:.0%}"
            color = "white" if (not np.isnan(val) and val > 0.6) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)

    return im


def plot_experiment_10(data: dict[str, Any], save_dir: str = "figures"):
    """Create and save heatmap and line-plot figures for Experiment 10."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    epsilons = data["epsilons"]
    model_names = data["model_names"]
    results_per_eps = data["results_per_eps"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes_flat = axes.flatten()

    for idx, eps in enumerate(epsilons):
        transfer_rates = results_per_eps[eps]["transfer_rates"]
        mat = _build_matrix(transfer_rates["PGD"], model_names)
        im = _plot_heatmap(mat, model_names, axes_flat[idx],
                           title=rf"$\varepsilon = {eps}$")

    fig.suptitle("PGD Adversarial Transfer Rate", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "exp10_heatmap_pgd.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for idx, atk in enumerate(["FGSM", "PGD"]):
        ax = axes[idx]
        for src in model_names:
            rates = []
            for eps in epsilons:
                tr = results_per_eps[eps]["transfer_rates"][atk][src]
                off_diag = [tr[tgt] for tgt in model_names if tgt != src]
                rates.append(np.nanmean(off_diag))
            ax.plot(epsilons, rates, LINE_STYLES[src], color=COLORS[src], label=src)

        ax.set_xlabel(r"Perturbation budget $\varepsilon$")
        ax.set_title(atk)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    axes[0].set_ylabel("Avg. transfer rate to other models")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "exp10_transfer_vs_eps.pdf"))
    plt.close(fig)

    print(f"Experiment 10 figures saved to '{save_dir}/'.")


if __name__ == "__main__":
    data = load_experiment_data("transferability")
    plot_experiment_10(data)
