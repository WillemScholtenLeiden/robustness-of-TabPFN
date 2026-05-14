"""Experiment 3: TabPFN vs standard MLP vs adversarially-trained MLP."""
import time
from typing import Any, Callable

import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from fgsm_attack import fgsm_attack, fgsm_attack_nn
from pgd_attack import pgd_linf_restarts, pgd_attack_nn
from raw_predict_with_grad import raw_predict_with_grad, enable_grad_raw_predict
from mlp import SimpleNN, train_standard_nn, train_adversarial_nn

from helpers import (
    _print_section,
    _print_elapsed,
    _save_experiment_data,
    create_n_synthetic_datasets,
    evaluate_multiple_epsilons,
    experiment_result,
)


def run_experiment_3(device: str = "cpu") -> dict[str, Any]:
    """Train TabPFN, MLP, and MLP-Adv; attack each with FGSM and PGD at multiple epsilons."""
    _print_section("EXPERIMENT 3: TabPFN vs RealMLP vs Adversarial RealMLP")
    t_start = time.perf_counter()

    n_datasets = 3
    datasets = create_n_synthetic_datasets(device=device, n_samples=525, test_size=25, n_datasets=n_datasets)
    epsilons = [0.01, 0.05, 0.1, 0.5, 1]
    steps = 10
    restarts = 5
    model_names = ["TabPFN", "RealMLP", "RealMLP-Adv"]

    per_model: dict[str, dict[str, Any]] = {name: {"per_dataset": []} for name in model_names}
    per_dataset: list[dict[str, dict[str, dict[float, experiment_result]]]] = []

    elapsed_per_dataset: list[float] = []

    for dataset_idx, (X_train, y_train, X_test, y_test) in enumerate(datasets):
        print(f"\n--- Dataset {dataset_idx + 1}/{n_datasets} ---")
        t_ds = time.perf_counter()

        clf_tabpfn = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            differentiable_input=True,
            device=device,
        )
        clf_tabpfn.fit(X_train, y_train)
        enable_grad_raw_predict(clf_tabpfn, raw_predict_with_grad)

        print("  Training standard RealMLP...")
        clf_mlp = train_standard_nn(
            X_train, y_train,
            epochs=100, lr=0.001, device=device
        )

        print("  Training adversarial RealMLP...")
        clf_mlp_adv = train_adversarial_nn(
            X_train, y_train,
            epochs=100, lr=0.001, eps=0.1, alpha=0.01, steps=10, device=device
        )

        models: dict[str, tuple[Any, Callable, Callable, dict[str, Any]]] = {
            "TabPFN": (clf_tabpfn, fgsm_attack, pgd_linf_restarts, {"steps": steps, "restarts": restarts}),
            "RealMLP": (clf_mlp, fgsm_attack_nn, pgd_attack_nn, {"steps": steps, "restarts": restarts}),
            "RealMLP-Adv": (clf_mlp_adv, fgsm_attack_nn, pgd_attack_nn, {"steps": steps, "restarts": restarts}),
        }

        dataset_block: dict[str, dict[str, dict[float, experiment_result]]] = {}

        for model_name, (clf, fgsm_fn, pgd_fn, pgd_kwargs) in models.items():
            print(f"  Evaluating {model_name}...")

            fgsm_by_eps: dict[float, experiment_result] = evaluate_multiple_epsilons(
                clf, X_test, y_test, fgsm_fn, epsilons
            )
            pgd_by_eps: dict[float, experiment_result] = evaluate_multiple_epsilons(
                clf, X_test, y_test, pgd_fn, epsilons, **pgd_kwargs
            )

            dataset_block[model_name] = {"fgsm": fgsm_by_eps, "pgd": pgd_by_eps}

            per_model[model_name]["per_dataset"].append({
                "dataset_idx": dataset_idx,
                "fgsm": fgsm_by_eps,
                "pgd": pgd_by_eps,
            })

        per_dataset.append(dataset_block)

        ds_elapsed = time.perf_counter() - t_ds
        elapsed_per_dataset.append(ds_elapsed)
        _print_elapsed(f"Dataset {dataset_idx + 1}", ds_elapsed)

    _print_section("PER-DATASET RESULTS: MODEL COMPARISON")

    for dataset_idx in range(n_datasets):
        print(f"\n=== Dataset {dataset_idx + 1} ===")
        for attack_type in ["fgsm", "pgd"]:
            print(f"\n  --- {attack_type.upper()} Attack Success Rates ---")
            print(f"  {'Epsilon':<10} {'TabPFN':<15} {'RealMLP':<15} {'RealMLP-Adv':<15}")
            print("  " + "-" * 55)

            for eps in epsilons:
                row = f"  {eps:<10.2f}"
                for name in model_names:
                    exp_res = per_dataset[dataset_idx][name][attack_type][eps]
                    sr = exp_res.calculate_success_rate()
                    row += f" {sr:.3f}         "
                print(row)

    total_elapsed = time.perf_counter() - t_start
    _print_elapsed("Experiment 3 total", total_elapsed)

    data = {
        "epsilons": epsilons,
        "n_datasets": n_datasets,
        "per_model": per_model,
        "per_dataset": per_dataset,
        "elapsed_total": total_elapsed,
        "elapsed_per_dataset": elapsed_per_dataset,
    }
    _save_experiment_data("comparison_with_nns", data)
    return data


if __name__ == "__main__":
    _device = "cuda"
    run_experiment_3(_device)
