"""
Plotting functions for Experiment 5: Runtime Analysis of Attacks on TabPFN.

Usage:
    data = run_experiment_5(device="cpu")
    plot_experiment_5(data, save_dir="figures")
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Any
from helpers import load_experiment_data
from plot_style import PLOT_STYLE

C_FGSM  = "#0072B2"
C_PGD   = "#D55E00"
C_TRAIN = "#009E73"
C_RATIO = "#CC79A7"


def _apply_style():
    mpl.rcParams.update(PLOT_STYLE)


def _mean_std(values_by_key: dict[int, list[float]], keys: list[int]):
    """Return arrays of means and stds for *keys* (in ms)."""
    means = np.array([np.mean(values_by_key[k]) * 1000 for k in keys])
    stds  = np.array([np.std(values_by_key[k])  * 1000 for k in keys])
    return means, stds


def plot_per_sample_runtime(data: dict[str, Any], ax: plt.Axes | None = None):
    """Line plot of per-sample FGSM and PGD runtime vs dataset size."""
    sizes = data["dataset_sizes"]
    fgsm_m, fgsm_s = _mean_std(data["fgsm_per_sample"], sizes)
    pgd_m,  pgd_s  = _mean_std(data["pgd_per_sample"],  sizes)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.errorbar(sizes, fgsm_m, yerr=fgsm_s, fmt="o-",
                color=C_FGSM, capsize=3, label="FGSM")
    ax.errorbar(sizes, pgd_m, yerr=pgd_s, fmt="s-",
                color=C_PGD, capsize=3, label=f"PGD ({data['pgd_steps']} steps, "
                                               f"{data['pgd_restarts']} restarts)")
    ax.set_xlabel("Dataset Size (samples)")
    ax.set_ylabel("Per-Sample Attack Time (ms)")
    ax.set_title("Attack Runtime vs. Dataset Size")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_xticks(sizes)

    if own_fig:
        fig.tight_layout()
    return ax


def plot_total_runtime(data: dict[str, Any], ax: plt.Axes | None = None):
    """Grouped bar chart of total FGSM vs PGD runtime per dataset size."""
    sizes = data["dataset_sizes"]
    fgsm_m, fgsm_s = _mean_std(data["fgsm_total"], sizes)
    pgd_m,  pgd_s  = _mean_std(data["pgd_total"],  sizes)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))

    x = np.arange(len(sizes))
    w = 0.35
    ax.bar(x - w / 2, fgsm_m, w, yerr=fgsm_s, color=C_FGSM,
           capsize=3, label="FGSM", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, pgd_m, w, yerr=pgd_s, color=C_PGD,
           capsize=3, label="PGD", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Dataset Size (samples)")
    ax.set_ylabel("Total Attack Time (ms)")
    ax.set_title("Total Runtime vs. Dataset Size")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")

    if own_fig:
        fig.tight_layout()
    return ax


def plot_speed_ratio(data: dict[str, Any], ax: plt.Axes | None = None):
    """Line plot showing PGD-to-FGSM runtime ratio vs dataset size."""
    sizes = data["dataset_sizes"]
    fgsm_m, _ = _mean_std(data["fgsm_per_sample"], sizes)
    pgd_m,  _ = _mean_std(data["pgd_per_sample"],  sizes)
    ratio = pgd_m / fgsm_m

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.plot(sizes, ratio, "D-", color=C_RATIO, markerfacecolor="white",
            markeredgewidth=1.2)
    ax.axhline(data["pgd_steps"] * data["pgd_restarts"], ls="--",
               color="0.5", lw=0.8, label="Theoretical ratio "
               rf"($steps \times restarts = {data['pgd_steps'] * data['pgd_restarts']}$)")
    ax.set_xlabel("Dataset Size (samples)")
    ax.set_ylabel("PGD / FGSM Runtime Ratio")
    ax.set_title("Relative Cost of PGD vs. FGSM")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_xticks(sizes)

    if own_fig:
        fig.tight_layout()
    return ax


def plot_train_time(data: dict[str, Any], ax: plt.Axes | None = None):
    """Bar chart of TabPFN training (fit) time vs dataset size."""
    sizes = data["dataset_sizes"]
    train_m, train_s = _mean_std(data["train_times"], sizes)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))

    x = np.arange(len(sizes))
    ax.bar(x, train_m, 0.5, yerr=train_s, color=C_TRAIN,
           capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Dataset Size (samples)")
    ax.set_ylabel("Fit Time (ms)")
    ax.set_title("TabPFN Training Time vs. Dataset Size")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)

    if own_fig:
        fig.tight_layout()
    return ax


def plot_experiment_5(data: dict[str, Any], save_dir: str = "figures"):
    """Create and save all Experiment 5 figures."""
    _apply_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    plot_per_sample_runtime(data, ax=axes[0, 0])
    plot_total_runtime(data, ax=axes[0, 1])
    plot_speed_ratio(data, ax=axes[1, 0])
    plot_train_time(data, ax=axes[1, 1])

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.15, 1.08, f"({label})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.5, h_pad=2.5)
    fig.savefig(os.path.join(save_dir, "exp5_combined.pdf"))
    plt.close(fig)

    print(f"All Experiment 5 figures saved to '{save_dir}/'.")

if __name__ == '__main__':
    data = load_experiment_data("data_size_scaling")

    print(data['elapsed_total'])

    plot_experiment_5(data)