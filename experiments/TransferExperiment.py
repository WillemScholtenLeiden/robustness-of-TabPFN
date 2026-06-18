import os
import time
import pickle
import torch
from tqdm import tqdm
import numpy as np
import gc
from pprint import pprint
from helpers.model_helpers import _train_models, _predict_target

class TransferExperiment:
    models = list()
    dataset_loaders = list()
    device = str()
    results = dict()
    exp_args = dict()
    total_run_time = float()

    def __init__(self, dataset_loaders: list, models: list, exp_args: dict = None):
        self.models = models
        self.dataset_loaders = dataset_loaders
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if exp_args is None:
            self.exp_args = {'eps': [0.25, 0.5, 1.0]}
        else:
            self.exp_args = exp_args

        self.results = {}
        self.total_run_time = 0.0

    def transfer_attack(self, eps: float, source_model: tuple, target_model: tuple,
                        x: torch.Tensor, y: torch.Tensor) -> bool | None:
        src_name, src_clf, attack_fn, attack_args = source_model
        result = attack_fn(src_clf, x, y, eps=eps, **attack_args)
        y_true = int(y.item()) if isinstance(y, torch.Tensor) else int(y)

        if result.pred_adv == y_true:
            return None

        x_adv = result.x_adv
        pred_target = _predict_target(target_model, x_adv)

        return pred_target != y_true

    def _init_dataset_results(self, dataset_idx: int):
        self.results[dataset_idx] = {}
        for eps in self.exp_args['eps']:
            self.results[dataset_idx][eps] = {}
            for (src_name, _, _, _) in self.models:
                self.results[dataset_idx][eps][src_name] = {}
                for (tgt_name, _, _, _) in self.models:
                    self.results[dataset_idx][eps][src_name][tgt_name] = []

    def perform_experiment(self):
        run_start = time.perf_counter()

        for dataset_idx, (loader, data_args) in enumerate(tqdm(self.dataset_loaders, desc='datasets')):
            self._init_dataset_results(dataset_idx)
            X_train, y_train, X_test, y_test = loader(device=self.device, **data_args)
            _train_models(self.models, X_train, y_train)

            for i in tqdm(range(len(X_test)), desc='samples', leave=False):
                x, y = X_test[i], y_test[i]
                y_true = int(y.item()) if isinstance(y, torch.Tensor) else int(y)

                for eps in self.exp_args['eps']:
                    for source_model in self.models:
                        src_name, src_clf, attack_fn, attack_args = source_model
                        result = attack_fn(src_clf, x, y, eps=eps, **attack_args)

                        if result.pred_adv == y_true:
                            for target_model in self.models:
                                self.results[dataset_idx][eps][src_name][target_model[0]].append(None)
                            continue

                        x_adv = result.x_adv
                        for target_model in self.models:
                            pred_target = _predict_target(target_model, x_adv)
                            self.results[dataset_idx][eps][src_name][target_model[0]].append(pred_target != y_true)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.total_run_time = time.perf_counter() - run_start


    def transfer_rate(self, dataset_idx: int, eps: float, source_name: str, target_name: str) -> float:
        outcomes = self.results[dataset_idx][eps][source_name][target_name]
        valid = [o for o in outcomes if o is not None]

        if len(valid) == 0:
            return float('nan')
        
        return sum(valid) / len(valid)

    def transfer_matrix(self, dataset_idx: int, eps: float) -> dict[str, dict[str, float]]:
        model_names = [m[0] for m in self.models]
        matrix = {}

        for src in model_names:
            matrix[src] = {}
            for tgt in model_names:
                matrix[src][tgt] = self.transfer_rate(dataset_idx, eps, src, tgt)

        return matrix
    
    def print_results(self):
        pprint(self.results)

    def save_results(self, path: str = 'results/transfer.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'results': self.results,
            'exp_args': self.exp_args,
            'total_run_time': self.total_run_time,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)