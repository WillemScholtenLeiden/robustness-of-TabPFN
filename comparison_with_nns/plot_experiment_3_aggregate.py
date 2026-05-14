"""
Aggregate plotting functions for Experiment 3.

Usage:
    data = load_experiment_data("comparison_with_nns")
    plot_experiment_3_aggregate(data, save_dir="figures")
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Any
from helpers import load_experiment_data
from plot_style import PLOT_STYLE

C_TABPFN  = "#0072B2"
C_MLP     = "#D55E00"
C_MLPADV  = "#009E73"

DISPLAY_NAMES = {
    "TabPFN": "TabPFN",
    "RealMLP": "MLP",
    "RealMLP-Adv": "MLP-Adv",
}

MODEL_COLORS = {
    "TabPFN": C_TABPFN,
    "RealMLP": C_MLP,
    "RealMLP-Adv": C_MLPADV,
}

MODEL_MARKERS = {
    "TabPFN": "o",
    "RealMLP": "s",
    "RealMLP-Adv": "^",
}


def _apply_style():
    mpl.rcParams.update(PLOT_STYLE)


def _compute_model_asr_stats(data: dict[str, Any], model_name: str, attack_type: str):
    """Compute mean and std of ASR across datasets for each epsilon."""
    epsilons = data["epsilons"]
    n_datasets = data["n_datasets"]

    means = []
    stds = []

    for eps in epsilons:
        asr_values = []
        for dataset_idx in range(n_datasets):
            exp_result = data["per_dataset"][dataset_idx][model_name][attack_type][eps]
            asr = exp_result.calculate_success_rate()
            if asr >= 0:
                asr_values.append(asr)

        if asr_values:
            means.append(np.mean(asr_values))
            stds.append(np.std(asr_values))
        else:
            means.append(0.0)
            stds.append(0.0)

    return np.array(means), np.array(stds)


def plot_fgsm_comparison(data: dict[str, Any], ax: plt.Axes | None = None):
    """Line plot comparing FGSM ASR across models (aggregated over datasets)."""
    epsilons = data["epsilons"]
    model_names = list(data["per_dataset"][0].keys())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for model_name in model_names:
        means, stds = _compute_model_asr_stats(data, model_name, "fgsm")
        ax.errorbar(epsilons, means, yerr=stds,
                    fmt=f"{MODEL_MARKERS[model_name]}-",
                    color=MODEL_COLORS[model_name],
                    capsize=3, label=DISPLAY_NAMES[model_name])

    ax.set_xlabel(r"Perturbation Budget ($\varepsilon$)")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("FGSM Attack: Model Comparison")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_ylim(-0.05, 1.05)

    if own_fig:
        fig.tight_layout()
    return ax


def plot_pgd_comparison(data: dict[str, Any], ax: plt.Axes | None = None):
    """Line plot comparing PGD ASR across models (aggregated over datasets)."""
    epsilons = data["epsilons"]
    model_names = list(data["per_dataset"][0].keys())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for model_name in model_names:
        means, stds = _compute_model_asr_stats(data, model_name, "pgd")
        ax.errorbar(epsilons, means, yerr=stds,
                    fmt=f"{MODEL_MARKERS[model_name]}-",
                    color=MODEL_COLORS[model_name],
                    capsize=3, label=DISPLAY_NAMES[model_name])

    ax.set_xlabel(r"Perturbation Budget ($\varepsilon$)")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("PGD Attack: Model Comparison")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_ylim(-0.05, 1.05)

    if own_fig:
        fig.tight_layout()
    return ax


def plot_grouped_bar_by_epsilon(data: dict[str, Any], attack_type: str = "fgsm",
                                 ax: plt.Axes | None = None):
    """Grouped bar chart showing ASR for each model at each epsilon."""
    epsilons = data["epsilons"]
    model_names = list(data["per_dataset"][0].keys())
    n_models = len(model_names)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))

    x = np.arange(len(epsilons))
    width = 0.8 / n_models

    for i, model_name in enumerate(model_names):
        means, stds = _compute_model_asr_stats(data, model_name, attack_type)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds,
               color=MODEL_COLORS[model_name], capsize=2,
               label=DISPLAY_NAMES[model_name], edgecolor="white", linewidth=0.5)

    ax.set_xlabel(r"Perturbation Budget ($\varepsilon$)")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title(f"{attack_type.upper()} Attack Success Rate by Model")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_ylim(0, 1.05)

    if own_fig:
        fig.tight_layout()
    return ax


def plot_model_robustness_comparison(data: dict[str, Any], ax: plt.Axes | None = None):
    """Bar chart comparing average ASR across all epsilons for each model."""
    model_names = list(data["per_dataset"][0].keys())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))

    fgsm_avgs = []
    pgd_avgs = []

    for model_name in model_names:
        fgsm_means, _ = _compute_model_asr_stats(data, model_name, "fgsm")
        pgd_means, _ = _compute_model_asr_stats(data, model_name, "pgd")
        fgsm_avgs.append(np.mean(fgsm_means))
        pgd_avgs.append(np.mean(pgd_means))

    x = np.arange(len(model_names))
    width = 0.35

    ax.bar(x - width/2, fgsm_avgs, width, label="FGSM",
           color="#0072B2", edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, pgd_avgs, width, label="PGD",
           color="#D55E00", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Attack Success Rate")
    ax.set_title("Overall Model Vulnerability")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[n] for n in model_names])
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_ylim(0, 1.05)

    if own_fig:
        fig.tight_layout()
    return ax


def plot_experiment_3_aggregate(data: dict[str, Any], save_dir: str = "figures"):
    """Create and save the aggregate 2x2 panel figure for Experiment 3."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    plot_fgsm_comparison(data, ax=axes[0, 0])
    plot_pgd_comparison(data, ax=axes[0, 1])
    plot_grouped_bar_by_epsilon(data, "fgsm", ax=axes[1, 0])
    plot_model_robustness_comparison(data, ax=axes[1, 1])

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.15, 1.08, f"({label})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.5, h_pad=2.5)
    fig.savefig(os.path.join(save_dir, "exp3_aggregate.pdf"))
    plt.close(fig)

    print(f"Experiment 3 aggregate figure saved to '{save_dir}/exp3_aggregate.pdf'.")


if __name__ == '__main__':
    data = load_experiment_data("comparison_with_nns")
    plot_experiment_3_aggregate(data)
