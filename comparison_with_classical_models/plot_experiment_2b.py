"""
Plotting functions for Experiment 2b: Per-Dataset Comparative Evaluation.

Usage:
    data = run_experiment_2b(device="cpu")
    plot_experiment_2b(data, save_dir="figures")
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Any
from helpers import load_experiment_data
from plot_style import PLOT_STYLE

C_TABPFN = "#0072B2"
C_LOGREG = "#D55E00"
C_SVM    = "#009E73"

MODEL_COLORS = {
    "TabPFN": C_TABPFN,
    "LogReg": C_LOGREG,
    "SVM": C_SVM,
}

MODEL_MARKERS = {
    "TabPFN": "o",
    "LogReg": "s",
    "SVM": "^",
}


def _apply_style():
    mpl.rcParams.update(PLOT_STYLE)


def _get_asr_for_dataset(data: dict[str, Any], dataset_idx: int, model_name: str,
                          attack_type: str) -> list[float]:
    """Get attack success rates for a specific dataset, model, and attack type."""
    epsilons = data["epsilons"]
    per_dataset = data["per_dataset"]

    asr_values = []
    for eps in epsilons:
        exp_result = per_dataset[dataset_idx][model_name][attack_type][eps]
        asr = exp_result.calculate_success_rate()
        asr_values.append(asr if asr >= 0 else 0.0)

    return asr_values


def plot_dataset_comparison(data: dict[str, Any], dataset_idx: int,
                            attack_type: str = "fgsm", ax: plt.Axes | None = None):
    """Line plot comparing models for a single dataset."""
    epsilons = data["epsilons"]
    model_names = list(data["per_dataset"][dataset_idx].keys())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for model_name in model_names:
        asr_values = _get_asr_for_dataset(data, dataset_idx, model_name, attack_type)
        ax.plot(epsilons, asr_values, f"{MODEL_MARKERS[model_name]}-",
                color=MODEL_COLORS[model_name], label=model_name)

    ax.set_xlabel(r"Perturbation Budget ($\varepsilon$)")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title(f"Dataset {dataset_idx + 1}: {attack_type.upper()} Attack")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_ylim(-0.05, 1.05)

    if own_fig:
        fig.tight_layout()
    return ax


def plot_all_datasets_fgsm(data: dict[str, Any], save_dir: str = "figures"):
    """Create separate figures for FGSM results on each dataset."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    n_datasets = data["n_datasets"]

    for dataset_idx in range(n_datasets):
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        plot_dataset_comparison(data, dataset_idx, "fgsm", ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"exp2b_dataset_{dataset_idx + 1}_fgsm.pdf"))
        plt.close(fig)


def plot_all_datasets_pgd(data: dict[str, Any], save_dir: str = "figures"):
    """Create separate figures for PGD results on each dataset."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    n_datasets = data["n_datasets"]

    for dataset_idx in range(n_datasets):
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        plot_dataset_comparison(data, dataset_idx, "pgd", ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"exp2b_dataset_{dataset_idx + 1}_pgd.pdf"))
        plt.close(fig)


def plot_grid_fgsm_pgd(data: dict[str, Any], save_dir: str = "figures"):
    """Create a grid figure showing FGSM and PGD for all datasets."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    n_datasets = data["n_datasets"]

    fig, axes = plt.subplots(n_datasets, 2, figsize=(7, 3 * n_datasets))

    for dataset_idx in range(n_datasets):
        plot_dataset_comparison(data, dataset_idx, "fgsm", axes[dataset_idx, 0])
        plot_dataset_comparison(data, dataset_idx, "pgd", axes[dataset_idx, 1])

    for idx, ax_row in enumerate(axes):
        ax_row[0].set_ylabel(f"Dataset {idx + 1}\nASR")

    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    fig.savefig(os.path.join(save_dir, "exp2b_grid.pdf"))
    plt.close(fig)


def plot_model_consistency(data: dict[str, Any], ax: plt.Axes | None = None):
    """Box plot showing ASR distribution across datasets for each model."""
    epsilons = data["epsilons"]
    n_datasets = data["n_datasets"]
    model_names = list(data["per_dataset"][0].keys())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 3.5))

    model_data = {name: [] for name in model_names}

    for dataset_idx in range(n_datasets):
        for model_name in model_names:
            for attack_type in ["fgsm", "pgd"]:
                for eps in epsilons:
                    exp_result = data["per_dataset"][dataset_idx][model_name][attack_type][eps]
                    asr = exp_result.calculate_success_rate()
                    if asr >= 0:
                        model_data[model_name].append(asr)

    positions = np.arange(len(model_names))
    bp = ax.boxplot([model_data[name] for name in model_names],
                    positions=positions, widths=0.6, patch_artist=True)

    for patch, model_name in zip(bp['boxes'], model_names):
        patch.set_facecolor(MODEL_COLORS[model_name])
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Model Vulnerability Distribution")
    ax.set_ylim(-0.05, 1.05)

    if own_fig:
        fig.tight_layout()
    return ax

def plot_experiment_2b(data: dict[str, Any], save_dir: str = "figures"):
    """Create and save all Experiment 2b figures."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    plot_grid_fgsm_pgd(data, save_dir)
    plot_all_datasets_fgsm(data, save_dir)
    plot_all_datasets_pgd(data, save_dir)

    print(f"All Experiment 2b figures saved to '{save_dir}/'.")


if __name__ == '__main__':
    data = load_experiment_data("comparison_with_classical_models")
    plot_experiment_2b(data)
