import torch
from experiments.EpsilonSweepExperiment import EpsilonSweepExperiment
from helpers.data_handlers import create_synthetic_dataset
from helpers.pgd_attack import pgd_linf_restarts, pgd_attack_sklearn
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

exp_args = {
    'eps': [0.01] + [i*0.1 for i in range(1, 11)]
}

data_args = {
    'random_state': 42,
    'n_features': 5
}

attack_args_FGSM = {
    'steps': 1, 
    'restarts': 1, 
    'seed': 42
}

attack_args_PGD = {
    'steps': 5, 
    'restarts': 5, 
    'seed': 42
}

dataset_loaders = []

# One garbage run to warm up the GPU and avoid outliers in timing
dataset_loaders.append((create_synthetic_dataset, {'random_state': 42, 'n_samples': 4, 'n_features': 2}))

for i in range(1, 5):
    dataset_loaders.append((create_synthetic_dataset, {'n_samples': i*24, **data_args}))

models = []

TabPFN_FGSM_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, differentiable_input=True)
TabPFN_FGSM = ('TabPFN_FGSM', TabPFN_FGSM_clf, pgd_linf_restarts, attack_args_FGSM)

TabPFN_PGD_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, differentiable_input=True)
TabPFN_PGD = ('TabPFN_PGD', TabPFN_PGD_clf, pgd_linf_restarts, attack_args_PGD)

models.append(TabPFN_FGSM)
models.append(TabPFN_PGD)

exp = EpsilonSweepExperiment(dataset_loaders, models, exp_args=exp_args, record_time=True)
exp.perform_experiment()
exp.print_results()
exp.save_results('results/data_timing.pkl')