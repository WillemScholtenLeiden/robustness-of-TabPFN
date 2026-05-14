"""
Robustness upper-bound distributions on Wisconsin Breast Cancer dataset.

For each test sample, binary-searches the minimum epsilon that causes
misclassification under FGSM and PGD for TabPFN, LogReg, and SVM.

PGD is evaluated with four configurations:
  (steps=5, restarts=5), (steps=10, restarts=5),
  (steps=5, restarts=10), (steps=10, restarts=10)

Additionally computes transferability distributions: for each test sample,
finds the minimum epsilon at which the LogReg/SVM adversarial example
also fools TabPFN.

"""

from __future__ import annotations

import gc
import time
from typing import Any, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from attacks_common import _predict_logits, _to_python_label
from fgsm_attack import fgsm_attack, fgsm_attack_sklearn
from pgd_attack import pgd_attack_sklearn
from raw_predict_with_grad import raw_predict_with_grad, enable_grad_raw_predict
from upper_bounds import find_upper_bound, find_upper_bound_pgd, find_upper_bound_sklearn, find_upper_bound_transfer
from helpers import create_breast_cancer_dataset, _print_section, _print_elapsed, _save_experiment_data


def _print_bounds_summary(label: str, bounds: list[float | None], n_test: int, eps_hi: float):
    """Print a one-line summary of robustness upper bounds."""
    n_robust = sum(1 for b in bounds if b is None)
    n_misclassified = sum(1 for b in bounds if b == 0.0)
    finite_bounds = [b for b in bounds if b is not None and b > 0.0]
    median_ub = float(np.median(finite_bounds)) if finite_bounds else float("nan")
    print(f"  [{label}] Samples: {n_test}  |  Already misclassified: {n_misclassified}  |  "
          f"Robust beyond {eps_hi}: {n_robust}  |  Median upper bound: {median_ub:.4f}")


PGD_CONFIGS = [
    (5, 5),
    (10, 5),
    (5, 10),
    (10, 10),
]


def _pgd_key(steps: int, restarts: int) -> str:
    """Return a dict key encoding a PGD configuration."""
    return f"pgd_s{steps}_r{restarts}"



def run_robustness_bc_dist_v2(device: str = "cpu") -> dict[str, Any]:
    """Compute per-sample robustness upper bounds for TabPFN, LogReg, and SVM."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    _print_section("ROBUSTNESS DISTRIBUTIONS v2: Wisconsin Breast Cancer Dataset")
    t_start = time.perf_counter()

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    eps_lo = 0.0
    eps_hi = 4.0
    tol = 1e-3

    X_train, y_train, X_test, y_test = create_breast_cancer_dataset(device=device)

    print("\n  Training TabPFN...")
    clf_tabpfn = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, differentiable_input=True, device=device,
    )
    clf_tabpfn.fit(X_train, y_train)
    enable_grad_raw_predict(clf_tabpfn, raw_predict_with_grad)

    print("  Training Logistic Regression...")
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    clf_lr = LogisticRegression(random_state=42, max_iter=1000)
    clf_lr.fit(X_train_np, y_train_np)

    print("  Training SVM...")
    clf_svm = SVC(kernel="linear", probability=True, random_state=42)
    clf_svm.fit(X_train_np, y_train_np)

    n_test = len(X_test)

    bounds_tabpfn_fgsm: list[float | None] = []
    bounds_lr_fgsm: list[float | None] = []
    bounds_svm_fgsm: list[float | None] = []
    bounds_transfer_lr_fgsm: list[float | None] = []
    bounds_transfer_svm_fgsm: list[float | None] = []

    bounds_tabpfn_pgd: dict[str, list[float | None]] = {_pgd_key(s, r): [] for s, r in PGD_CONFIGS}
    bounds_lr_pgd: dict[str, list[float | None]] = {_pgd_key(s, r): [] for s, r in PGD_CONFIGS}
    bounds_svm_pgd: dict[str, list[float | None]] = {_pgd_key(s, r): [] for s, r in PGD_CONFIGS}
    bounds_transfer_lr_pgd: dict[str, list[float | None]] = {_pgd_key(s, r): [] for s, r in PGD_CONFIGS}
    bounds_transfer_svm_pgd: dict[str, list[float | None]] = {_pgd_key(s, r): [] for s, r in PGD_CONFIGS}

    for i in range(n_test):
        print(f"  Sample {i + 1}/{n_test}", end="\r")

        xi = X_test[i]
        yi = y_test[i]

        bounds_tabpfn_fgsm.append(find_upper_bound(
            clf_tabpfn, xi, yi, eps_lo=eps_lo, eps_hi=eps_hi, tol=tol,
        ))
        bounds_lr_fgsm.append(find_upper_bound_sklearn(
            clf_lr, xi, yi, attack_fn=fgsm_attack_sklearn,
            eps_lo=eps_lo, eps_hi=eps_hi, tol=tol,
        ))
        bounds_svm_fgsm.append(find_upper_bound_sklearn(
            clf_svm, xi, yi, attack_fn=fgsm_attack_sklearn,
            eps_lo=eps_lo, eps_hi=eps_hi, tol=tol,
        ))
        bounds_transfer_lr_fgsm.append(find_upper_bound_transfer(
            clf_lr, fgsm_attack_sklearn, clf_tabpfn, xi, yi,
            eps_lo=eps_lo, eps_hi=eps_hi, tol=tol, device=device,
        ))
        bounds_transfer_svm_fgsm.append(find_upper_bound_transfer(
            clf_svm, fgsm_attack_sklearn, clf_tabpfn, xi, yi,
            eps_lo=eps_lo, eps_hi=eps_hi, tol=tol, device=device,
        ))

        for pgd_steps, pgd_restarts in PGD_CONFIGS:
            key = _pgd_key(pgd_steps, pgd_restarts)

            bounds_tabpfn_pgd[key].append(find_upper_bound_pgd(
                clf_tabpfn, xi, yi, eps_lo=eps_lo, eps_hi=eps_hi, tol=tol,
                pgd_steps=pgd_steps, pgd_restarts=pgd_restarts, seed=i,
            ))
            bounds_lr_pgd[key].append(find_upper_bound_sklearn(
                clf_lr, xi, yi, attack_fn=pgd_attack_sklearn,
                eps_lo=eps_lo, eps_hi=eps_hi, tol=tol,
                steps=pgd_steps, restarts=pgd_restarts, seed=i,
            ))
            bounds_svm_pgd[key].append(find_upper_bound_sklearn(
                clf_svm, xi, yi, attack_fn=pgd_attack_sklearn,
                eps_lo=eps_lo, eps_hi=eps_hi, tol=tol,
                steps=pgd_steps, restarts=pgd_restarts, seed=i,
            ))
            bounds_transfer_lr_pgd[key].append(find_upper_bound_transfer(
                clf_lr, pgd_attack_sklearn, clf_tabpfn, xi, yi,
                eps_lo=eps_lo, eps_hi=eps_hi, tol=tol, device=device,
                steps=pgd_steps, restarts=pgd_restarts, seed=i,
            ))
            bounds_transfer_svm_pgd[key].append(find_upper_bound_transfer(
                clf_svm, pgd_attack_sklearn, clf_tabpfn, xi, yi,
                eps_lo=eps_lo, eps_hi=eps_hi, tol=tol, device=device,
                steps=pgd_steps, restarts=pgd_restarts, seed=i,
            ))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    _print_section("STANDARD ROBUSTNESS BOUNDS")

    for model_label, fgsm_bounds, pgd_bounds in [
        ("TabPFN", bounds_tabpfn_fgsm, bounds_tabpfn_pgd),
        ("Logistic Regression", bounds_lr_fgsm, bounds_lr_pgd),
        ("SVM", bounds_svm_fgsm, bounds_svm_pgd),
    ]:
        print(f"  --- {model_label} ---")
        _print_bounds_summary("FGSM", fgsm_bounds, n_test, eps_hi)
        for pgd_steps, pgd_restarts in PGD_CONFIGS:
            key = _pgd_key(pgd_steps, pgd_restarts)
            _print_bounds_summary(f"PGD (s={pgd_steps}, r={pgd_restarts})", pgd_bounds[key], n_test, eps_hi)

    _print_section("TRANSFER ROBUSTNESS BOUNDS (source adv. examples -> TabPFN)")

    for source_label, fgsm_bounds, pgd_bounds in [
        ("LogReg -> TabPFN", bounds_transfer_lr_fgsm, bounds_transfer_lr_pgd),
        ("SVM -> TabPFN", bounds_transfer_svm_fgsm, bounds_transfer_svm_pgd),
    ]:
        print(f"  --- {source_label} ---")
        _print_bounds_summary("FGSM", fgsm_bounds, n_test, eps_hi)
        for pgd_steps, pgd_restarts in PGD_CONFIGS:
            key = _pgd_key(pgd_steps, pgd_restarts)
            _print_bounds_summary(f"PGD (s={pgd_steps}, r={pgd_restarts})", pgd_bounds[key], n_test, eps_hi)

    total_elapsed = time.perf_counter() - t_start
    _print_elapsed("Robustness BC dist v2 total", total_elapsed)

    data = {
        "upper_bounds_fgsm": bounds_tabpfn_fgsm,
        "upper_bounds_lr_fgsm": bounds_lr_fgsm,
        "upper_bounds_svm_fgsm": bounds_svm_fgsm,
        "upper_bounds_transfer_lr_fgsm": bounds_transfer_lr_fgsm,
        "upper_bounds_transfer_svm_fgsm": bounds_transfer_svm_fgsm,
        "upper_bounds_pgd": bounds_tabpfn_pgd,
        "upper_bounds_lr_pgd": bounds_lr_pgd,
        "upper_bounds_svm_pgd": bounds_svm_pgd,
        "upper_bounds_transfer_lr_pgd": bounds_transfer_lr_pgd,
        "upper_bounds_transfer_svm_pgd": bounds_transfer_svm_pgd,
        "pgd_configs": PGD_CONFIGS,
        "eps_lo": eps_lo,
        "eps_hi": eps_hi,
        "tol": tol,
        "elapsed_total": total_elapsed,
    }
    _save_experiment_data("robustness_bc_dist_v2", data)
    return data


if __name__ == "__main__":
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {_device}")
    run_robustness_bc_dist_v2(device=_device)
