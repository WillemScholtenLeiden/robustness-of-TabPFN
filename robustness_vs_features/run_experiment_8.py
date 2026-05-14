"""Experiment 8: Robustness upper bounds across feature dimensionalities."""
import time
from typing import Any

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from raw_predict_with_grad import raw_predict_with_grad, enable_grad_raw_predict
from upper_bounds import find_upper_bound, find_upper_bound_pgd

from helpers import (
    _print_section,
    _print_elapsed,
    _save_experiment_data,
    create_n_synthetic_datasets,
)


def run_experiment_8(
    device: str = "cpu",
) -> dict[str, Any]:
    """Compute per-sample robustness upper bounds for multiple feature dimensions."""
    _print_section("EXPERIMENT 8: Robustness Upper Bounds across Feature Dimensions")
    t_start = time.perf_counter()

    n_samples = 100
    feature_counts = [5, 15, 25]
    n_datasets = 2
    test_size = 0.25
    eps_lo = 0.0
    eps_hi = 4.0
    tol = 1e-3

    results_by_nfeat: dict[int, dict[str, list[list[float | None]]]] = {}

    for n_features in feature_counts:
        print(f"\n=== n_features = {n_features} ===")

        datasets = create_n_synthetic_datasets(
            n_samples=n_samples,
            n_features=n_features,
            n_datasets=n_datasets,
            test_size=test_size,
            device=device,
        )

        all_upper_bounds_fgsm: list[list[float | None]] = []
        all_upper_bounds_pgd: list[list[float | None]] = []

        for ds_idx, (X_train, y_train, X_test, y_test) in enumerate(datasets):
            print(f"\n--- Dataset {ds_idx + 1}/{n_datasets} ---")

            clf = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                differentiable_input=True,
                device=device,
            )
            clf.fit(X_train, y_train)
            enable_grad_raw_predict(clf, raw_predict_with_grad)

            n_test = len(X_test)
            bounds_fgsm: list[float | None] = []
            bounds_pgd: list[float | None] = []

            for i in range(n_test):
                print(f"  Sample {i + 1}/{n_test}", end="\r")
                ub_fgsm = find_upper_bound(
                    clf, X_test[i], y_test[i],
                    eps_lo=eps_lo, eps_hi=eps_hi, tol=tol,
                )
                bounds_fgsm.append(ub_fgsm)

                ub_pgd = find_upper_bound_pgd(
                    clf, X_test[i], y_test[i],
                    eps_lo=eps_lo, eps_hi=eps_hi, tol=tol,
                )
                bounds_pgd.append(ub_pgd)

            for label, bounds in [("FGSM", bounds_fgsm), ("PGD", bounds_pgd)]:
                n_robust = sum(1 for b in bounds if b is None)
                n_misclassified = sum(1 for b in bounds if b == 0.0)
                finite_bounds = [b for b in bounds if b is not None and b > 0.0]
                median_ub = float(np.median(finite_bounds)) if finite_bounds else float("nan")
                print(f"  [{label}] Samples: {n_test}  |  Already misclassified: {n_misclassified}  |  "
                      f"Robust beyond {eps_hi}: {n_robust}  |  Median upper bound: {median_ub:.4f}")

            all_upper_bounds_fgsm.append(bounds_fgsm)
            all_upper_bounds_pgd.append(bounds_pgd)

        results_by_nfeat[n_features] = {
            "all_upper_bounds_fgsm": all_upper_bounds_fgsm,
            "all_upper_bounds_pgd": all_upper_bounds_pgd,
        }

    total_elapsed = time.perf_counter() - t_start
    _print_elapsed("Experiment 8 total", total_elapsed)

    data = {
        "results_by_nfeat": results_by_nfeat,
        "feature_counts": feature_counts,
        "n_samples": n_samples,
        "n_datasets": n_datasets,
        "test_size": test_size,
        "eps_lo": eps_lo,
        "eps_hi": eps_hi,
        "tol": tol,
        "elapsed_total": total_elapsed,
    }
    _save_experiment_data("robustness_vs_features", data)
    return data


if __name__ == "__main__":
    _device = "cpu"
    run_experiment_8(_device)
