"""
Combined ECDF plot for robustness v2 experiments.

Shows two panels:
  - Left:  PGD transfer attacks (s=10, r=10) — all models incl. transfer
  - Right: TabPFN PGD distributions — all 4 PGD configs (TabPFN only)

Usage:
    python plot_robustness_bc_v2_combined.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

from helpers import load_experiment_data
from plot_style import PLOT_STYLE

STYLE = {**PLOT_STYLE, "legend.fontsize": 8}

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent

PGD_KEY = "pgd_s10_r10"

COLORS = {
    "TabPFN": "#0072B2",
    "LogReg": "#D55E00",
    "SVM": "#009E73",
    "LogReg\u2192TabPFN": "#CC79A7",
    "SVM\u2192TabPFN": "#E69F00",
}

TABPFN_PGD_COLORS = {
    "pgd_s5_r5":   "#6BAED6",
    "pgd_s10_r5":  "#3182BD",
    "pgd_s5_r10":  "#08519C",
    "pgd_s10_r10": "#0072B2",
}


def _compute_ecdf(bounds: list[float | None], eps_hi: float) -> tuple[np.ndarray, np.ndarray]:
    resolved = np.array([b if b is not None else eps_hi for b in bounds])
    sorted_vals = np.sort(resolved)
    n = len(sorted_vals)
    ecdf_y = np.arange(1, n + 1) / n
    return sorted_vals, ecdf_y


def _step(ax, bounds, eps_hi, *, color, label, **kwargs):
    xs, ys = _compute_ecdf(bounds, eps_hi)
    ax.step(xs, ys, where="post", color=color, label=label, **kwargs)


def plot_combined(data: dict[str, Any], save_dir: Path = FIGURES_DIR):
    """Create and save a two-panel ECDF figure (PGD transfer + TabPFN configs)."""
    mpl.rcParams.update(STYLE)
    save_dir.mkdir(parents=True, exist_ok=True)

    eps_hi = data["eps_hi"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

    ax = axes[0]

    _step(ax, data["upper_bounds_pgd"][PGD_KEY], eps_hi,
          color=COLORS["TabPFN"], label="TabPFN",
          linewidth=2.0, linestyle="-")

    _step(ax, data["upper_bounds_lr_pgd"][PGD_KEY], eps_hi,
          color=COLORS["LogReg"], label="LogReg",
          linewidth=1.5, linestyle="-")
    _step(ax, data["upper_bounds_svm_pgd"][PGD_KEY], eps_hi,
          color=COLORS["SVM"], label="SVM",
          linewidth=1.5, linestyle="-")

    _step(ax, data["upper_bounds_transfer_lr_pgd"][PGD_KEY], eps_hi,
          color=COLORS["LogReg\u2192TabPFN"],
          label="LogReg\u2192TabPFN",
          linewidth=1.5, linestyle="--")

    _step(ax, data["upper_bounds_transfer_svm_pgd"][PGD_KEY], eps_hi,
          color=COLORS["SVM\u2192TabPFN"],
          label="SVM\u2192TabPFN",
          linewidth=1.5, linestyle="--")

    ax.set_xlabel(r"Perturbation budget $\varepsilon$")
    ax.set_ylabel("Fraction of samples misclassified")
    ax.set_title("PGD Transfer (steps=10, restarts=10)")
    ax.set_xlim(0, eps_hi)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=1)

    ax = axes[1]

    _step(ax, data["upper_bounds_fgsm"], eps_hi,
          color=COLORS["TabPFN"], label="TabPFN FGSM",
          linewidth=2.0, linestyle="-")

    pgd_styles = {
        "pgd_s5_r5":   (":", 1.2),
        "pgd_s10_r5":  ("--", 1.2),
        "pgd_s5_r10":  ("-.", 1.2),
        "pgd_s10_r10": ("-", 1.5),
    }
    for key, (ls, lw) in pgd_styles.items():
        s, r = key.split("_")[1], key.split("_")[2]
        _step(ax, data["upper_bounds_pgd"][key], eps_hi,
              color=TABPFN_PGD_COLORS[key],
              label=f"TabPFN PGD ({s},{r})",
              linewidth=lw, linestyle=ls)

    ax.set_xlabel(r"Perturbation budget $\varepsilon$")
    ax.set_title("TabPFN PGD Configurations")
    ax.set_xlim(0, eps_hi)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=1)

    fig.tight_layout()
    out = save_dir / "robustness_bc_v2_combined.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    pkl_path = RESULTS_DIR / "robustness_bc_dist_v2.pkl"
    if not pkl_path.exists():
        print(f"Results file not found: {pkl_path}")
        sys.exit(1)

    data = load_experiment_data("robustness_bc_dist_v2", results_dir=RESULTS_DIR)
    plot_combined(data)
