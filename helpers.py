"""Shared helpers for running and evaluating adversarial-robustness experiments.

Provides dataset creation (synthetic and Breast Cancer), per-sample attack
evaluation loops, result aggregation, console formatting, and pickle-based
I/O for experiment data.
"""
import io
import pickle
import time
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class experiment_result:
    """Stores all per-sample AttackResults for one (attack, epsilon) setting."""
    def __init__(self):
        self.parameters: dict[str, Any] = {}
        self.results: list[Any] = []
        self.runtime: float = -1.0

    def calculate_success_rate(self) -> float:
        """Fraction of correctly-classified samples that were fooled."""
        successes = 0
        total = 0

        for result in self.results:
            if result.pred_nat == result.y_true:
                total += 1
                if result.pred_adv != result.y_true:
                    successes += 1

        if total == 0:
            return -1.0
        return successes / total

    def calculate_adversarial_accuracy(self) -> float:
        """Fraction of all samples correctly classified under attack."""
        if len(self.results) == 0:
            return -1.0

        correct = 0
        for result in self.results:
            if result.pred_adv == result.y_true:
                correct += 1
        return correct / len(self.results)

    def calculate_clean_accuracy(self) -> float:
        """Fraction of all samples correctly classified without attack."""
        if len(self.results) == 0:
            return -1.0

        correct = 0
        for result in self.results:
            if result.pred_nat == result.y_true:
                correct += 1
        return correct / len(self.results)

    def get_scores(self) -> dict[str, float]:
        """Return a dict with ASR, adversarial accuracy, and clean accuracy."""
        return {
            "asr": self.calculate_success_rate(),
            "aac": self.calculate_adversarial_accuracy(),
            "cac": self.calculate_clean_accuracy(),
        }

def create_synthetic_dataset(
    n_samples: int = 100,
    n_features: int = 5,
    test_size: float = 0.25,
    random_state: int = 42,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a binary classification dataset, standardise, and return tensors."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    return X_train, y_train, X_test, y_test


def create_n_synthetic_datasets(
    n_samples: int = 100,
    n_features: int = 5,
    n_datasets: int = 3,
    test_size: float = 0.25,
    device: str = "cpu",
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Create multiple synthetic datasets with sequential random seeds."""
    datasets = []
    for seed in range(n_datasets):
        dataset = create_synthetic_dataset(
            n_samples=n_samples,
            n_features=n_features,
            test_size=test_size,
            random_state=42 + seed,
            device=device,
        )
        datasets.append(dataset)
    return datasets


def create_breast_cancer_dataset(
    test_size: float = 0.25,
    random_state: int = 42,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the Wisconsin Breast Cancer dataset, standardise, and return tensors."""
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    return X_train, y_train, X_test, y_test

def _to_tensor_on_device(
    arr: Any, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Convert an array-like to a torch tensor on the specified device."""
    if not isinstance(arr, torch.Tensor):
        return torch.tensor(arr, device=device)
    return arr.to(device)

def evaluate_attack(
    clf: Any,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    attack_fn: Callable,
    eps: float,
    **attack_kwargs: Any,
) -> experiment_result:
    """Run a single attack function on every test sample at a given epsilon."""
    results = []
    start_time = time.perf_counter()

    for i in range(len(X_test)):
        x_nat = X_test[i]
        y_true = y_test[i]

        result = attack_fn(clf, x_nat, y_true, eps=eps, **attack_kwargs)

        results.append(result)

    elapsed_time = time.perf_counter() - start_time

    exp_result = experiment_result()
    exp_result.results = results
    exp_result.parameters = {"eps": eps, **attack_kwargs}
    exp_result.runtime = elapsed_time

    return exp_result

def evaluate_multiple_epsilons(
    clf: Any,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    attack_fn: Callable,
    epsilons: list[float],
    **attack_kwargs: Any,
) -> dict[float, experiment_result]:
    """Run ``evaluate_attack`` for each epsilon and return a dict keyed by epsilon."""
    results: dict[float, experiment_result] = {}

    for eps in tqdm(epsilons, leave=False):
        results[eps] = evaluate_attack(
            clf, X_test, y_test, attack_fn, eps, **attack_kwargs
        )

    return results


def _aggregate(
    records: list[dict[str, float]], key: str
) -> tuple[float, float]:
    """Return (mean, std) of ``key`` across a list of score dicts."""
    values = [r[key] for r in records]
    return float(np.mean(values)), float(np.std(values))


def _print_section(title: str) -> None:
    """Print a section header banner to stdout."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def _print_elapsed(label: str, elapsed: float) -> None:
    """Print a human-readable elapsed time (seconds or minutes)."""
    if elapsed < 60:
        print(f"  [{label}] elapsed: {elapsed:.2f}s")
    else:
        mins, secs = divmod(elapsed, 60)
        print(f"  [{label}] elapsed: {int(mins)}m {secs:.2f}s")


RESULTS_DIR = Path("results")

def _save_experiment_data(name: str, data: List, results_dir: Path = RESULTS_DIR) -> Path:
    """Pickle experiment data to ``results/<name>.pkl``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Results saved to {path}")
    return path




class _CpuUnpickler(pickle.Unpickler):
    """Custom unpickler that maps CUDA tensors to CPU."""
    def find_class(self, module, name):
        # Remap old pickle references from before AttackResult was centralised
        if name == 'AttackResult' and module in ('fgsm_attack', 'pgd_attack'):
            module = 'attacks_common'
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def load_experiment_data(name: str, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Load a pickled experiment from ``results/<name>.pkl``, mapping CUDA tensors to CPU."""
    path = results_dir / f"{name}.pkl"
    with open(path, "rb") as f:
        data = _CpuUnpickler(f).load()
    print(f"Results loaded from {path}")
    return data