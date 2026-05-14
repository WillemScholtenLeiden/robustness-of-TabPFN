"""
Plotting functions for Experiment 8: Robustness across Feature Dimensions.

Usage:
    data = load_experiment_data("robustness_vs_features")
    plot_experiment_8(data, save_dir="figures")
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Any
from helpers import load_experiment_data
from plot_style import PLOT_STYLE

COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
]


def _apply_style():
    mpl.rcParams.update(PLOT_STYLE)


def _compute_ecdf(bounds: list[float | None], eps_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the empirical CDF of upper bounds (None values clamped to eps_hi)."""
    resolved = np.array([b if b is not None else eps_hi for b in bounds])
    sorted_vals = np.sort(resolved)
    n = len(sorted_vals)
    ecdf_y = np.arange(1, n + 1) / n
    return sorted_vals, ecdf_y


def _plot_dataset_cdf(
    feature_counts: list[int],
    bounds_by_nfeat: dict[int, list[float | None]],
    eps_hi: float,
    ax: plt.Axes,
    title: str,
) -> None:
    """Plot one ECDF curve per feature count on the given axes."""
    for feat_idx, n_feat in enumerate(feature_counts):
        bounds = bounds_by_nfeat[n_feat]
        xs, ys = _compute_ecdf(bounds, eps_hi)
        color = COLORS[feat_idx % len(COLORS)]
        ax.step(xs, ys, where="post", color=color,
                linewidth=1.5, label=f"{n_feat} features")

    ax.set_xlabel(r"Perturbation budget $\varepsilon$")
    ax.set_title(title)
    ax.set_xlim(0, eps_hi)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7", loc="lower right")


def plot_experiment_8(data: dict[str, Any], save_dir: str = "figures"):
    """Create and save a 2x2 grid of robustness ECDFs (datasets x attacks)."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    eps_hi = data["eps_hi"]
    feature_counts = data["feature_counts"]
    results_by_nfeat = data["results_by_nfeat"]

    n_datasets = 2  # rows: Dataset 1, Dataset 2
    attack_keys = [
        ("all_upper_bounds_fgsm", "FGSM"),
        ("all_upper_bounds_pgd", "PGD"),
    ]

    fig, axes = plt.subplots(n_datasets, 2, figsize=(11, 8), sharey=True, sharex=True)

    for ds_idx in range(n_datasets):
        for col_idx, (key, attack_name) in enumerate(attack_keys):
            bounds_by_nfeat = {}
            for n_feat in feature_counts:
                bounds_by_nfeat[n_feat] = results_by_nfeat[n_feat][key][ds_idx]

            _plot_dataset_cdf(
                feature_counts,
                bounds_by_nfeat,
                eps_hi,
                axes[ds_idx, col_idx],
                title=f"{attack_name} — Dataset {ds_idx + 1}",
            )

            if col_idx == 0:
                axes[ds_idx, col_idx].set_ylabel("Fraction of samples misclassified")
            else:
                axes[ds_idx, col_idx].set_ylabel("")

    for col in range(2):
        axes[0, col].set_xlabel("")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "exp8_robustness_by_features.pdf"))
    plt.close(fig)

    print(f"Experiment 8 figures saved to '{save_dir}/'.")


if __name__ == "__main__":
    data = load_experiment_data("robustness_vs_features")
    plot_experiment_8(data)