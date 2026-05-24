# Investigating Local Robustness of TabPFN on Binary Classification Tasks

This repository contains the code for evaluating the adversarial robustness of [TabPFN](https://github.com/PriorLabs/TabPFN) (a tabular prior-data fitted network) under L-infinity threat models, comparing it against classical ML models and neural networks.

## Project Structure

```
.
├── helpers/                        # Core modules
│   ├── AttackResult.py             # Frozen dataclass for attack outputs
│   ├── pgd_attack.py              # PGD L_infty attacks for TabPFN, sklearn, and NN models
│   ├── transfer_attack.py         # Cross-model adversarial transfer logic
│   ├── enable_grad_raw_predict.py # Gradient-enabled inference patch for TabPFN
│   ├── data_handlers.py           # Dataset loading (synthetic + UCI) and preprocessing
│   ├── mlp.py                     # Standard and adversarially-trained MLP classifiers
│   ├── model_helpers.py           # Model prediction and training utilities
│   └── attack_helpers.py          # Logit prediction helpers and tensor conversion
│
├── experiments/                    # Experiment frameworks
│   ├── EpsilonSweepExperiment.py  # Attack success rates across epsilon budgets
│   ├── TransferExperiment.py      # Cross-model transfer attack evaluation
│   └── KBinarySearchExperiment.py # Binary search for minimum adversarial epsilon
│
├── tests/                          # Pytest regression tests
│   ├── test_core.py
│   └── conftest.py
│
├── classical_comparison.py         # TabPFN vs LogReg vs SVM (FGSM/PGD)
├── nn_comparison.py                # TabPFN vs MLP vs Adversarial MLP (FGSM/PGD)
├── feature_experiment.py           # ERD vs feature dimensionality
├── time_data.py                    # Attack runtime vs training data size
├── time_features.py                # Runtime scaling across features
├── four_transfer.py                # Transfer attacks on Sonar, Ionosphere, Spambase, Haberman
├── wisconsin_transfer.py           # Transfer attacks on Breast Cancer dataset
└── erd_winsconsin.py               # ERD on Breast Cancer
```

## Experiments

| Experiment | Script | Description |
|---|---|---|
| Classical Comparison | `classical_comparison.py` | FGSM attack success rates across TabPFN, Logistic Regression, and SVM on synthetic data |
| NN Comparison | `nn_comparison.py` | FGSM/PGD comparison of TabPFN, standard MLP, and adversarially-trained MLP |
| Runtime vs Data Size | `time_data.py` | Per-sample and total attack runtime as training set size increases |
| Runtime & ERD vs Features | `feature_experiment.py`, `time_features.py` | Attack runtime & ERD as feature dimensionality increases |
| Robustness Bounds | `erd_winsconsin.py` | Per-sample robustness bounds via binary search on Breast Cancer |
| Transferability | `four_transfer.py`, `wisconsin_transfer.py` | Cross-model adversarial transferability on UCI datasets |

## Attack Methods

All attacks operate under the **L-infinity** threat model:

- **FGSM** (Fast Gradient Sign Method) — single-step attack (`steps=1, restarts=1`)
- **PGD** (Projected Gradient Descent) — multi-step, multi-restart attack with step size `alpha = 2 * eps / steps`

### Attacking TabPFN

TabPFN runs inference under `torch.inference_mode()`, which disables gradient computation. The `enable_grad_raw_predict.py` module patches this to allow gradient flow through the model, enabling white-box adversarial attacks via TabPFN's `differentiable_input=True` mode.

## Datasets

| Dataset | Source |
|---|---|
| Synthetic | `sklearn.datasets.make_classification` |
| Breast Cancer | scikit-learn |
| Sonar | UCI #151 |
| Ionosphere | UCI #52 |
| Banknote | UCI #267 |
| Haberman | UCI #43 |
| Spambase | UCI #94 |

All datasets use a 75/25 stratified train/test split with standard scaling fitted on the training set.

## Models

- **TabPFN v2** — Prior-data fitted network for tabular data
- **Logistic Regression** — scikit-learn `LogisticRegression`
- **SVM** — scikit-learn `LinearSVC` / `SVC`
- **Standard MLP** — 2-layer fully connected network (64 → 32 → 2)
- **Adversarial MLP** — Same architecture trained with PGD adversarial training

## Requirements

- Python 3.x
- PyTorch
- TabPFN v2
- scikit-learn
- NumPy
- Matplotlib
- tqdm
- ucimlrepo
- pytest (for tests)

## Usage

Run any experiment script directly:

```bash
python classical_comparison.py
python nn_comparison.py
python four_transfer.py
```

Run the test suite:

```bash
pytest tests/
```

> **Note:** Tests marked `@pytest.mark.slow` (TabPFN-dependent) are excluded by default. Run them with `pytest --runslow`.

## License

This project is part of a Bachelors's thesis in Data Science and AI.
