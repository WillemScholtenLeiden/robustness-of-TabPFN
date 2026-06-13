import torch
from experiments.TransferExperiment import TransferExperiment
from helpers.data_handlers import load_sonar_dataset, load_ionosphere_dataset, load_spambase_dataset, load_haberman_dataset
from helpers.pgd_attack import pgd_linf_restarts, pgd_attack_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_args = {
    'eps_low': 0.0, 
    'eps_high': 4.0, 
    'tol': 1e-3
}

data_args = {}

attack_args = {
    'steps': 1, 
    'restarts': 1, 
    'seed': 42
}

dataset_loaders = [
    (load_sonar_dataset, data_args),
    (load_ionosphere_dataset, data_args),
    (load_spambase_dataset, data_args),
    (load_haberman_dataset, data_args)
]

models = []

TabPFN_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, differentiable_input=True, device=_device)
TabPFN = ('TabPFN_FGSM', TabPFN_clf, pgd_linf_restarts, attack_args)

SVM_clf = SVC(kernel='linear', probability=True, random_state=42)
SVM = ('SVM_FGSM', SVM_clf, pgd_attack_sklearn, attack_args)

LogReg_clf = LogisticRegression(random_state=42, max_iter=1000)
LogReg = ('LogisticRegression_FGSM', LogReg_clf, pgd_attack_sklearn, attack_args)

models.append(TabPFN)
models.append(SVM)
models.append(LogReg)

exp = TransferExperiment(dataset_loaders, models, exp_args=exp_args)
exp.perform_experiment()
exp.print_results()
exp.save_results('results/four_transfer.pkl')
