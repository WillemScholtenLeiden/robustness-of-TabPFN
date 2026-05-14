"""Experiment 10: Adversarial transferability on Wisconsin Breast Cancer data."""
import time
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from fgsm_attack import fgsm_attack, fgsm_attack_sklearn
from pgd_attack import pgd_linf_restarts, pgd_attack_sklearn
from raw_predict_with_grad import raw_predict_with_grad, enable_grad_raw_predict

from helpers import (
    _print_section,
    _print_elapsed,
    _save_experiment_data,
    create_breast_cancer_dataset,
)

MODEL_NAMES = ["TabPFN", "LogReg", "SVM"]


def _predict_sklearn(clf, x: torch.Tensor) -> int:
    """Predict a single sample with a sklearn classifier."""
    x_np = x.detach().cpu().numpy().reshape(1, -1)
    return int(clf.predict(x_np)[0])


def _predict_tabpfn(clf, x: torch.Tensor) -> int:
    """Predict a single sample with a TabPFN classifier."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    with torch.no_grad():
        logits = clf._raw_predict(x, return_logits=True)
    return int(logits.argmax(dim=1).item())


def run_experiment_10(
    device: str = "cpu",
    epsilons: list[float] | None = None,
) -> dict[str, Any]:
    """Craft adversarial examples on each model and measure cross-model transfer rates."""
    _print_section("EXPERIMENT 10: Adversarial Transferability on Breast Cancer")
    t_start = time.perf_counter()

    if epsilons is None:
        epsilons = [0.05, 0.1, 0.25, 0.5, 1.0]

    X_train, y_train, X_test, y_test = create_breast_cancer_dataset(device=device)
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()

    print("\n  Training TabPFN...")
    clf_tabpfn = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        differentiable_input=True,
        device=device,
    )
    clf_tabpfn.fit(X_train, y_train)
    enable_grad_raw_predict(clf_tabpfn, raw_predict_with_grad)

    print("  Training Logistic Regression...")
    clf_lr = LogisticRegression(random_state=42, max_iter=1000)
    clf_lr.fit(X_train_np, y_train_np)

    print("  Training SVM...")
    clf_svm = SVC(kernel="linear", probability=True, random_state=42)
    clf_svm.fit(X_train_np, y_train_np)

    clfs = {"TabPFN": clf_tabpfn, "LogReg": clf_lr, "SVM": clf_svm}

    attack_fns = {
        "TabPFN": {
            "FGSM": lambda clf, x, y, eps: fgsm_attack(clf, x, y, eps=eps),
            "PGD": lambda clf, x, y, eps: pgd_linf_restarts(clf, x, y, eps=eps, steps=20, restarts=10),
        },
        "LogReg": {
            "FGSM": lambda clf, x, y, eps: fgsm_attack_sklearn(clf, x, y, eps=eps),
            "PGD": lambda clf, x, y, eps: pgd_attack_sklearn(clf, x, y, eps=eps, steps=20, restarts=10),
        },
        "SVM": {
            "FGSM": lambda clf, x, y, eps: fgsm_attack_sklearn(clf, x, y, eps=eps),
            "PGD": lambda clf, x, y, eps: pgd_attack_sklearn(clf, x, y, eps=eps, steps=20, restarts=10),
        },
    }

    predict_fns = {
        "TabPFN": lambda x: _predict_tabpfn(clf_tabpfn, x),
        "LogReg": lambda x: _predict_sklearn(clf_lr, x),
        "SVM": lambda x: _predict_sklearn(clf_svm, x),
    }

    n_test = len(X_test)

    results_per_eps = {}

    for eps in epsilons:
        print(f"\n  === eps = {eps} ===")
        counts = {
            atk: {s: {t: 0 for t in MODEL_NAMES} for s in MODEL_NAMES}
            for atk in ["FGSM", "PGD"]
        }
        n_source_success = {
            atk: {s: 0 for s in MODEL_NAMES}
            for atk in ["FGSM", "PGD"]
        }
        n_correct = {s: 0 for s in MODEL_NAMES}

        for i in range(n_test):
            print(f"  Sample {i + 1}/{n_test}", end="\r")
            x_i = X_test[i]
            y_i = y_test[i]
            y_true = int(y_i.item())

            clean_preds = {name: predict_fns[name](x_i) for name in MODEL_NAMES}
            for name in MODEL_NAMES:
                if clean_preds[name] == y_true:
                    n_correct[name] += 1

            for atk_name in ["FGSM", "PGD"]:
                for source in MODEL_NAMES:
                    if clean_preds[source] != y_true:
                        continue

                    result = attack_fns[source][atk_name](clfs[source], x_i, y_i, eps)

                    if result.pred_adv == y_true:
                        continue

                    n_source_success[atk_name][source] += 1
                    x_adv = result.x_adv
                    if x_adv.ndim == 2:
                        x_adv = x_adv.squeeze(0)

                    for target in MODEL_NAMES:
                        pred_target = predict_fns[target](x_adv)
                        if pred_target != y_true:
                            counts[atk_name][source][target] += 1

        transfer_rates = {}
        for atk_name in ["FGSM", "PGD"]:
            transfer_rates[atk_name] = {}
            for source in MODEL_NAMES:
                transfer_rates[atk_name][source] = {}
                n_succ = n_source_success[atk_name][source]
                for target in MODEL_NAMES:
                    if n_succ > 0:
                        transfer_rates[atk_name][source][target] = counts[atk_name][source][target] / n_succ
                    else:
                        transfer_rates[atk_name][source][target] = float("nan")

            print(f"\n  Transfer rates ({atk_name}, eps={eps}):")
            print(f"  {'Source → Target':<20}", end="")
            for t in MODEL_NAMES:
                print(f"{t:>10}", end="")
            print(f"{'  (n_succ)':>10}")
            for s in MODEL_NAMES:
                print(f"  {s:<20}", end="")
                for t in MODEL_NAMES:
                    rate = transfer_rates[atk_name][s][t]
                    print(f"{rate:>10.1%}", end="")
                print(f"{n_source_success[atk_name][s]:>10d}")

        results_per_eps[eps] = {
            "transfer_rates": transfer_rates,
            "counts": counts,
            "n_source_success": n_source_success,
            "n_correct": n_correct,
        }

    total_elapsed = time.perf_counter() - t_start
    _print_elapsed("Experiment 10 total", total_elapsed)

    data = {
        "epsilons": epsilons,
        "model_names": MODEL_NAMES,
        "results_per_eps": results_per_eps,
        "n_test": n_test,
        "elapsed_total": total_elapsed,
    }
    _save_experiment_data("transferability", data)
    return data


if __name__ == "__main__":
    _device = "cuda"
    run_experiment_10(_device)
