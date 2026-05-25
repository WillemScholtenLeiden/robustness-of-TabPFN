import torch
from experiments.KBinarySearchExperiment import KBinarySearchExperiment
from helpers.data_handlers import load_breast_cancer_dataset
from helpers.pgd_attack import pgd_linf_restarts, pgd_attack_sklearn
from helpers.transfer_attack import transfer_attack
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def left_panel():
    exp_args = {
        'eps_low': 0.0, 
        'eps_high': 4.0, 
        'tol': 1e-3
    }

    data_args = {}

    attack_args = {
        'steps': 10, 
        'restarts': 10, 
        'seed': 42
    }

    dataset_loaders = [(load_breast_cancer_dataset, data_args)]

    TabPFN_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, differentiable_input=True, device=_device)
    TabPFN = ('TabPFN', TabPFN_clf, pgd_linf_restarts, attack_args)

    SVM_clf = SVC(kernel='linear', probability=True, random_state=42)
    SVM = ('SVM', SVM_clf, pgd_attack_sklearn, attack_args)

    LogReg_clf = LogisticRegression(random_state=42, max_iter=1000)
    LogReg = ('LogisticRegression', LogReg_clf, pgd_attack_sklearn, attack_args)

    LogReg_TabPFN_args = {
        'source_clf': LogReg_clf,
        'source_attack_fn': pgd_attack_sklearn,
        'source_attack_args': attack_args,
        'target_clf': TabPFN_clf,
        'target_name': 'LogReg_TabPFN',
    }
    LogReg_TabPFN = ('LogReg_TabPFN', TabPFN_clf, transfer_attack, LogReg_TabPFN_args)

    SVM_TabPFN_args = {
        'source_clf': SVM_clf,
        'source_attack_fn': pgd_attack_sklearn,
        'source_attack_args': attack_args,
        'target_clf': TabPFN_clf,
        'target_name': 'SVM_TabPFN',
    }
    SVM_TabPFN = ('SVM_TabPFN', TabPFN_clf, transfer_attack, SVM_TabPFN_args)

    models = [TabPFN, SVM, LogReg, LogReg_TabPFN, SVM_TabPFN]

    exp = KBinarySearchExperiment(dataset_loaders, models, exp_args=exp_args)
    exp.perform_experiment()
    exp.print_results()
    exp.save_results('results/left_panel_erd_winsconsin.pkl')

def right_panel():
    exp_args = {
        'eps_low': 0.0, 
        'eps_high': 4.0, 
        'tol': 1e-3
    }

    data_args = {}

    attack_args = {
        'seed': 42
    }

    dataset_loaders = [(load_breast_cancer_dataset, data_args)]

    TabPFN_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, differentiable_input=True, device=_device)
    TabPFN_FGSM = ('TabPFN_FGSM', TabPFN_clf, pgd_linf_restarts, {'steps': 1, 'restarts': 1, **attack_args})
    TabPFN_5_5 = ('TabPFN_5_5', TabPFN_clf, pgd_linf_restarts, {'steps': 5, 'restarts': 5, **attack_args})
    TabPFN_10_5 = ('TabPFN_10_5', TabPFN_clf, pgd_linf_restarts, {'steps': 10, 'restarts': 5, **attack_args})
    TabPFN_5_10 = ('TabPFN_5_10', TabPFN_clf, pgd_linf_restarts, {'steps': 5, 'restarts': 10, **attack_args})
    TabPFN_10_10 = ('TabPFN_10_10', TabPFN_clf, pgd_linf_restarts, {'steps': 10, 'restarts': 10, **attack_args})

    models = [TabPFN_FGSM, TabPFN_5_5, TabPFN_10_5, TabPFN_5_10, TabPFN_10_10]

    exp = KBinarySearchExperiment(dataset_loaders, models, exp_args=exp_args)
    exp.perform_experiment()
    exp.print_results()
    exp.save_results('results/right_panel_erd_winsconsin.pkl')

    
if __name__ == "__main__":
    print("Left Panel:")
    left_panel()
    print("\nRight Panel:")
    right_panel()