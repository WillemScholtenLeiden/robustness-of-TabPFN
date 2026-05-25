import torch
from experiments.EpsilonSweepExperiment import EpsilonSweepExperiment
from helpers.data_handlers import create_synthetic_dataset
from helpers.pgd_attack import pgd_linf_restarts, pgd_attack_nn
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from helpers.mlp import StandardMLPClassifier, AdversarialMLPClassifier

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_args = {
    'eps': [0.01] + [i*0.1 for i in range(1, 11)]
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

all_attack_args = [('FGSM', attack_args_FGSM), ('PGD', attack_args_PGD)]

dataset_loaders = []

for i in range(5):
    dataset_loaders.append((create_synthetic_dataset, {'random_state': i+42}))

models = []

for attack_name, attack_args in all_attack_args:
    TabPFN_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, differentiable_input=True, device=_device)
    TabPFN = ('TabPFN_' + attack_name, TabPFN_clf, pgd_linf_restarts, attack_args)

    clf_mlp = StandardMLPClassifier(epochs=100, lr=0.001, device=_device)
    MLP = ('MLP_' + attack_name, clf_mlp, pgd_attack_nn, attack_args)

    clf_mlp_adv = AdversarialMLPClassifier(
        epochs=100, lr=0.001, eps=0.1, alpha=0.01, steps=10, device=_device
    )
    MLP_adv = ('MLP_Adv_' + attack_name, clf_mlp_adv, pgd_attack_nn, attack_args)

    models.append(TabPFN)
    models.append(MLP)
    models.append(MLP_adv)

exp = EpsilonSweepExperiment(dataset_loaders, models, exp_args=exp_args, device=_device)
exp.perform_experiment()
exp.print_results()
exp.save_results('results/nn_comparison.pkl')