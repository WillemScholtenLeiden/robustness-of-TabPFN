import torch
from experiments.KBinarySearchExperiment import KBinarySearchExperiment
from helpers.data_handlers import create_synthetic_dataset
from helpers.pgd_attack import pgd_linf_restarts, pgd_attack_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier, ModelVersion

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_args = {
    'eps_low': 0.0, 
    'eps_high': 4.0, 
    'tol': 1e-3
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

for seed in range(42, 44):
    for n_features in [5, 15, 25]:
        dataset_loaders.append(
            (create_synthetic_dataset, {
                    'n_samples': 100, 
                    'n_features': n_features, 
                    'random_state': seed
                })
        )

models = []

for attack_name, attack_args in all_attack_args:
    TabPFN_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, differentiable_input=True, device=_device)
    TabPFN = ('TabPFN_' + attack_name, TabPFN_clf, pgd_linf_restarts, attack_args)

    SVM_clf = SVC(kernel='linear', probability=True, random_state=42)
    SVM = ('SVM_' + attack_name, SVM_clf, pgd_attack_sklearn, attack_args)

    LogReg_clf = LogisticRegression(random_state=42, max_iter=1000)
    LogReg = ('LogisticRegression_' + attack_name, LogReg_clf, pgd_attack_sklearn, attack_args)

    models.append(TabPFN)
    models.append(SVM)
    models.append(LogReg)

exp = KBinarySearchExperiment(dataset_loaders, models, exp_args=exp_args)
exp.perform_experiment()
exp.print_results()
exp.save_results('results/feature_experiment.pkl')