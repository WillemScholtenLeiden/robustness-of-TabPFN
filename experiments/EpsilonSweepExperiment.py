import os
import time
import pickle
import torch
import tqdm
import gc
import numpy as np
from pprint import pprint
from helpers.model_helpers import _train_models
from helpers.AttackResult import AttackResult

class EpsilonSweepExperiment:
    exp_args = dict()
    dataset_loaders = list()
    models = list()
    device = str()
    results = dict()
    timings = dict()
    record_time = False
    total_run_time = float()

    def __init__(self, dataset_loaders: list, models: list, exp_args: dict = None, device: str = None, record_time: bool = False):
        self.dataset_loaders = dataset_loaders
        self.models = models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        self.exp_args = {'eps': [i*0.1 for i in range(1, 11)]} if exp_args is None else exp_args
        self.results = {}
        self.record_time = record_time
        self.timings = {}
        self.total_run_time = 0.0

    def evaluate_attack(self, model, eps: float, X_test: torch.Tensor, y_test: torch.Tensor):
        results = []
        per_sample_times = []
        total_start = time.perf_counter() if self.record_time else None

        for i in range(len(X_test)):
            x = X_test[i]
            y = y_test[i]

            sample_start = time.perf_counter() if self.record_time else None
            result = model[2](model[1], x, y, eps=eps, **model[3])
            if self.record_time:
                per_sample_times.append(time.perf_counter() - sample_start)
            results.append(result)

        if self.record_time:
            total_time = time.perf_counter() - total_start
            return results, {'per_sample_times': per_sample_times, 'total_time': total_time}
        return results

    def evaluate_multiple_epsilons(self, model, epss: list, X_test: torch.Tensor, y_test: torch.Tensor):
        results = {}
        timings = {}

        for eps in epss:
            out = self.evaluate_attack(model, eps, X_test, y_test)
            if self.record_time:
                results[eps], timings[eps] = out
            else:
                results[eps] = out

        if self.record_time:
            return results, timings
        return results

    def perform_experiment(self):
        run_start = time.perf_counter()
        
        for dataset_idx, (loader, data_args) in enumerate(tqdm.tqdm(self.dataset_loaders, desc='datasets')):
            self.results[dataset_idx] = {name: {} for (name, _, _, _) in self.models}
            if self.record_time:
                self.timings[dataset_idx] = {name: {} for (name, _, _, _) in self.models}
            X_train, y_train, X_test, y_test = loader(device=self.device, **data_args)
            _train_models(self.models, X_train, y_train)

            for (model_name, model, attack_fn, attack_args) in self.models:
                out = self.evaluate_multiple_epsilons(
                    (model_name, model, attack_fn, attack_args), self.exp_args['eps'], X_test, y_test
                )
                if self.record_time:
                    self.results[dataset_idx][model_name], self.timings[dataset_idx][model_name] = out
                else:
                    self.results[dataset_idx][model_name] = out

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.total_run_time = time.perf_counter() - run_start

    def print_results(self):
        pprint(self.results)

    def save_results(self, path: str = 'results/epsilon_sweep.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'results': self.results,
            'exp_args': self.exp_args,
            'total_run_time': self.total_run_time,
        }
        if self.record_time:
            payload['timings'] = self.timings
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
