"""Experiment 6: Attack runtime scaling with feature dimensionality.

Measures per-sample and total wall-clock time of FGSM and PGD attacks
on TabPFN as the number of input features increases, complementing
the dataset-size scaling analysis in Experiment 5.
"""
import time
from typing import Any

import numpy as np
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from fgsm_attack import fgsm_attack
from pgd_attack import pgd_linf_restarts
from raw_predict_with_grad import raw_predict_with_grad, enable_grad_raw_predict

from helpers import _print_section, create_synthetic_dataset, _print_elapsed, _save_experiment_data, \
    load_experiment_data


def run_experiment_6(
    device: str = "cpu",
) -> dict[str, Any]:
    """Benchmark FGSM and PGD attack runtime across increasing feature counts."""
    _print_section("EXPERIMENT 6: Runtime Analysis vs Number of Features")
    t_start = time.perf_counter()

    n_samples = 60
    feature_sizes = [2, 10, 20, 30, 40, 50]
    test_size = 0.25
    eps = 0.5
    pgd_steps = 5
    pgd_restarts = 5
    n_repeats = 3

    fgsm_per_sample: dict[int, list[float]] = {f: [] for f in feature_sizes}
    pgd_per_sample: dict[int, list[float]] = {f: [] for f in feature_sizes}
    fgsm_total: dict[int, list[float]] = {f: [] for f in feature_sizes}
    pgd_total: dict[int, list[float]] = {f: [] for f in feature_sizes}
    train_times: dict[int, list[float]] = {f: [] for f in feature_sizes}

    for n_features in feature_sizes:
        print(f"\n--- Number of Features: {n_features} ---")

        for repeat in range(n_repeats):
            X_train, y_train, X_test, y_test = create_synthetic_dataset(
                n_samples=n_samples,
                n_features=n_features,
                test_size=test_size,
                random_state=42,
                device=device,
            )
            n_test = len(X_test)

            t0 = time.perf_counter()
            clf = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                differentiable_input=True,
                device=device,
            )
            clf.fit(X_train, y_train)
            enable_grad_raw_predict(clf, raw_predict_with_grad)
            train_times[n_features].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            for i in range(n_test):
                fgsm_attack(clf, X_test[i], y_test[i], eps=eps)
            elapsed = time.perf_counter() - t0
            fgsm_total[n_features].append(elapsed)
            fgsm_per_sample[n_features].append(elapsed / n_test)

            t0 = time.perf_counter()
            for i in range(n_test):
                pgd_linf_restarts(
                    clf, X_test[i], y_test[i],
                    eps=eps, steps=pgd_steps, restarts=pgd_restarts,
                )
            elapsed = time.perf_counter() - t0
            pgd_total[n_features].append(elapsed)
            pgd_per_sample[n_features].append(elapsed / n_test)

        fm = np.mean(fgsm_per_sample[n_features]) * 1000
        pm = np.mean(pgd_per_sample[n_features]) * 1000
        print(f"  FGSM per-sample: {fm:.2f} ms  |  PGD per-sample: {pm:.2f} ms"
              f"  |  Ratio: {pm / fm:.2f}x")

    _print_section("RUNTIME VS FEATURES SUMMARY")

    print(f"\n{'Features':<10} {'FGSM (ms)':<18} {'PGD (ms)':<18} {'Ratio':<10}")
    print("-" * 56)
    for f in feature_sizes:
        fm = np.mean(fgsm_per_sample[f]) * 1000
        pm = np.mean(pgd_per_sample[f]) * 1000
        print(f"{f:<10} {fm:<18.2f} {pm:<18.2f} {pm / fm:<10.2f}x")

    total_elapsed = time.perf_counter() - t_start
    _print_elapsed("Experiment 6 total", total_elapsed)

    data = {
        "feature_sizes": feature_sizes,
        "n_samples": n_samples,
        "fgsm_per_sample": fgsm_per_sample,
        "pgd_per_sample": pgd_per_sample,
        "fgsm_total": fgsm_total,
        "pgd_total": pgd_total,
        "train_times": train_times,
        "pgd_steps": pgd_steps,
        "pgd_restarts": pgd_restarts,
        "eps": eps,
        "n_repeats": n_repeats,
        "elapsed_total": total_elapsed,
    }
    _save_experiment_data("feature_dim_scaling", data)
    return data

if __name__ == "main":
    _device = "cuda"

    run_experiment_6(_device)