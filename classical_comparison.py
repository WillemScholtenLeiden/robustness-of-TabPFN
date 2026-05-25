import torch
from experiments.EpsilonSweepExperiment import EpsilonSweepExperiment
from helpers.data_handlers import create_synthetic_dataset
from helpers.pgd_attack import pgd_linf_restarts, pgd_attack_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

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

dataset_loaders = []

for i in range(5):
    dataset_loaders.append((create_synthetic_dataset, {'random_state': i+42}))

TabPFN_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, differentiable_input=True, device=_device)
TabPFN_FGSM = ('TabPFN_FGSM', TabPFN_clf, pgd_linf_restarts, attack_args_FGSM)
TabPFN_PGD = ('TabPFN_PGD', TabPFN_clf, pgd_linf_restarts, attack_args_PGD)

SVM_clf = SVC(kernel='linear', probability=True, random_state=42)
SVM_FGSM = ('SVM_FGSM', SVM_clf, pgd_attack_sklearn, attack_args_FGSM)
SVM_PGD = ('SVM_PGD', SVM_clf, pgd_attack_sklearn, attack_args_PGD)

LogReg_clf = LogisticRegression(random_state=42, max_iter=1000)
LogReg_FGSM = ('LogisticRegression_FGSM', LogReg_clf, pgd_attack_sklearn, attack_args_FGSM)
LogReg_PGD = ('LogisticRegression_PGD', LogReg_clf, pgd_attack_sklearn, attack_args_PGD)

models = [TabPFN_FGSM, TabPFN_PGD, SVM_FGSM, SVM_PGD, LogReg_FGSM, LogReg_PGD]

exp = EpsilonSweepExperiment(dataset_loaders, models, exp_args=exp_args, device=_device)
exp.perform_experiment()
exp.print_results()
exp.save_results('results/classical_comparison.pkl')