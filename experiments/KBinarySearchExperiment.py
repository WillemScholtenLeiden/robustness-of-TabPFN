import os
import time
import pickle
import torch
import tqdm
import gc
import numpy as np
import matplotlib as mpl
from pprint import pprint
from helpers.model_helpers import _train_models
from AttackResults import AttackResults

class KBinarySearchExperiment:
    k = int()
    exp_args = dict()
    dataset_loaders = list()
    models = list()
    device = str()
    results = dict()
    total_run_time = float()

    def __init__(self, dataset_loaders: list, models: list, exp_args: dict = None, k: int = 2):
        self.k = k
        self.dataset_loaders = dataset_loaders
        self.models = models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if exp_args is not None:
            self.exp_args = exp_args
        else:
            self.exp_args = {'eps_low': 0.0, 'eps_high': 4.0, 'tol': 1e-3}

        self.results = {}
        self.total_run_time = 0.0

    def perform_k_binary_search(self, model: tuple, x: torch.Tensor, y: torch.Tensor, eps_low: float = 0.0,
                                eps_high: float = 4.0, tol: float = 1e-3,) -> float | None:
        result_nat = model[2](model[1], x, y, eps=0.0, **model[3])
        if result_nat.pred_nat != y:
            return 0.0

        result_hi = model[2](model[1], x, y, eps=eps_high, **model[3])
        if result_hi.pred_adv == y:
            return None

        while (eps_high - eps_low) > tol:
            for i in range(1, self.k):
                eps_mid = i * (eps_low + eps_high) / self.k
                result = model[2](model[1], x, y, eps=eps_mid, **model[3])
                flipped = result.pred_adv != y
                del result
                if flipped:
                    eps_high = eps_mid
                    break
                else:
                    eps_low = eps_mid

        return eps_high

    def perform_experiment(self):
        run_start = time.perf_counter()
        
        for dataset_idx, (loader, data_args) in enumerate(tqdm(self.dataset_loaders, desc=f'datasets')):
            self.results[dataset_idx] = {name: [] for (name, _, _, _) in self.models}
            X_train, y_train, X_test, y_test = loader(device=self.device, **data_args)
            _train_models(self.models, X_train, y_train)

            for i in tqdm(range(len(X_test)), desc=f'samples', leave=False):
                x, y = X_test[i], y_test[i]

                for (model_name, model, attack_fn, attack_args) in self.models:
                    self.results[dataset_idx][model_name].append(self.perform_k_binary_search((model_name, model, attack_fn, attack_args), x, y, **self.exp_args))

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.total_run_time = time.perf_counter() - run_start

    @staticmethod
    def _compute_ecdf(bounds: list, eps_hi: float):
        resolved = np.array([b if b is not None else eps_hi for b in bounds])
        sorted_vals = np.sort(resolved)
        n = len(sorted_vals)
        ecdf_y = np.arange(1, n + 1) / n

        return sorted_vals, ecdf_y

    def make_panel(self, dataset_idx: int = 0, selection: list = None, STYLE=None, eps_hi: float = None):
        if STYLE is not None:
            mpl.rcParams.update(STYLE)
        if selection is None:
            selection = list(self.results[dataset_idx].keys())
        if eps_hi is None:
            eps_hi = self.exp_args.get('eps_high', 4.0)

        fig, ax = mpl.subplots(1, 1, figsize=(5.5, 4))

        for model_name in selection:
            bounds = self.results[dataset_idx][model_name]
            xs, ys = self._compute_ecdf(bounds, eps_hi)
            ax.step(xs, ys, where="post", linewidth=1.5, label=model_name)

        ax.set_xlabel(r"Perturbation budget $\varepsilon$")
        ax.set_ylabel("Fraction of samples misclassified")
        ax.set_xlim(0, eps_hi)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")

        return fig, ax

    def print_results(self):
        pprint(self.results)

    def save_results(self, path: str = 'results/k_binary_search.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'results': self.results,
            'exp_args': self.exp_args,
            'k': self.k,
            'total_run_time': self.total_run_time,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
