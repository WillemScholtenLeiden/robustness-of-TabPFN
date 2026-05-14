# Investigating Local Robustness of TabPFN on Binary Classification Tasks

Experiment code for a thesis investigating the adversarial robustness of [TabPFN](https://github.com/automl/TabPFN) (a tabular prior-data fitted network) compared to classical machine learning models and standard neural networks. All attacks operate in the L-infinity threat model using FGSM ([Goodfellow et al., 2015](https://arxiv.org/abs/1412.6557)) and PGD ([Madry et al., 2018](https://arxiv.org/abs/1706.06083)).

## Project Structure

```
.
├── attacks_common.py             # AttackResult dataclass and shared helpers
├── fgsm_attack.py                # FGSM attacks for TabPFN, PyTorch NNs, and sklearn
├── pgd_attack.py                 # PGD attacks for TabPFN, PyTorch NNs, and sklearn
├── upper_bounds.py               # Binary-search routines for minimum adversarial epsilon
├── raw_predict_with_grad.py      # Gradient-enabled inference patch for TabPFN
├── helpers.py                    # Dataset creation, evaluation loops, I/O utilities
├── plot_style.py                 # Shared matplotlib rcParams for thesis figures
│
├── comparison_with_classical_models/
│   ├── run_experiment_2b.py      # Exp 2b: TabPFN vs LogReg vs SVM (synthetic)
│   ├── plot_experiment_2b.py     # Per-dataset figures
│   └── plot_experiment_2b_aggregate.py  # Aggregated 2x2 panel figure
│
├── comparison_with_nns/
│   ├── run_experiment_3.py       # Exp 3: TabPFN vs MLP vs MLP-Adv (synthetic)
│   ├── plot_experiment_3.py      # Per-dataset figures
│   └── plot_experiment_3_aggregate.py  # Aggregated 2x2 panel figure
│
├── data_size_scaling/
│   ├── run_experiment_5.py       # Exp 5: Attack runtime vs training-set size
│   └── plot_experiment_5.py      # Runtime scaling figures
│
├── feature_dim_scaling/
│   ├── run_experiment_6.py       # Exp 6: Attack runtime vs feature dimensionality
│   └── plot_experiment_6.py      # Runtime scaling figures
│
├── robustness_vs_features/
│   ├── run_experiment_8.py       # Exp 8: Robustness bounds across feature dimensions
│   └── plot_experiment_8.py      # ECDF grid figures
│
├── transferability/
│   ├── run_experiment_10.py      # Exp 10: Adversarial transferability (Breast Cancer)
│   └── plot_experiment_10.py     # Transfer-rate heatmaps and line plots
│
└── winsconsin_edc/
    ├── run_experiment.py         # Robustness upper-bound distributions (Breast Cancer)
    └── plot_experiment.py        # Combined ECDF panels
```

Each experiment directory contains a `results/` folder (pickled experiment data) and a `figures/` folder (generated PDFs) after execution.

## Experiments

| # | Name | Description |
|---|------|-------------|
| 2b | Classical Model Comparison | FGSM/PGD attack success rates for TabPFN, Logistic Regression, and SVM on synthetic data across multiple datasets and epsilon values |
| 3 | Neural Network Comparison | Same evaluation for TabPFN, a standard MLP, and an adversarially-trained MLP (PGD-AT) |
| 5 | Runtime vs Data Size | Per-sample and total attack runtime as training-set size increases |
| 6 | Runtime vs Features | Per-sample and total attack runtime as feature dimensionality increases |
| 8 | Robustness vs Features | Robustness upper bounds across feature dimensionalities (5, 15, 25 features) |
| 10 | Transferability | Cross-model adversarial transferability between TabPFN, LogReg, and SVM |
| -- | Robustness Distributions | Per-sample robustness upper bounds for TabPFN, LogReg, and SVM on Breast Cancer with FGSM, PGD, and transfer attacks |

## Shared Modules

### `attacks_common.py`

Defines the `AttackResult` dataclass returned by all attack functions, along with shared helpers (`_predict_logits`, `_to_numpy_1d`, `_to_python_label`).

### `fgsm_attack.py`

Single-step L-infinity adversarial perturbation: `x_adv = x + eps * sign(grad_x L)`.

| Function | Target Model |
|----------|-------------|
| `fgsm_attack()` | TabPFN (differentiable `_raw_predict`) |
| `fgsm_attack_nn()` | Any `torch.nn.Module` producing logits |
| `fgsm_attack_sklearn()` | LogisticRegression, LinearSVC, SVC (linear kernel) |

### `pgd_attack.py`

Multi-step, multi-restart L-infinity attack with step size `alpha = 2 * eps / steps`.

| Function | Target Model |
|----------|-------------|
| `pgd_linf_restarts()` | TabPFN (differentiable `_raw_predict`) |
| `pgd_attack_nn()` | Any `torch.nn.Module` producing logits |
| `pgd_attack_sklearn()` | LogisticRegression, LinearSVC, SVC (linear kernel) |

### `upper_bounds.py`

Binary-search routines that find the minimum epsilon causing misclassification for a given sample. Each function returns the threshold epsilon, `0.0` if the sample is already misclassified, or `None` if it remains robust beyond the search range.

| Function | Attack | Model |
|----------|--------|-------|
| `find_upper_bound()` | FGSM | TabPFN |
| `find_upper_bound_pgd()` | PGD | TabPFN |
| `find_upper_bound_sklearn()` | Any | sklearn linear classifiers |
| `find_upper_bound_transfer()` | Any | Cross-model (source -> TabPFN) |

### `raw_predict_with_grad.py`

TabPFN's default `_raw_predict` runs under `torch.inference_mode()`, which disables autograd. This module provides `raw_predict_with_grad`, a drop-in replacement that keeps the computation graph alive, and `enable_grad_raw_predict`, which applies it via monkey-patching:

```python
from raw_predict_with_grad import raw_predict_with_grad, enable_grad_raw_predict

clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, differentiable_input=True)
clf.fit(X_train, y_train)
enable_grad_raw_predict(clf, raw_predict_with_grad)
```

### `helpers.py`

Dataset creation (synthetic and Breast Cancer), per-sample attack evaluation loops, result aggregation, console formatting, and pickle-based I/O. Includes a custom unpickler that maps CUDA tensors to CPU and handles backward-compatible deserialization of `AttackResult`.

### `plot_style.py`

Exports `PLOT_STYLE`, a shared matplotlib rcParams dict used by all plotting scripts to produce consistent thesis-quality figures.

## Key Metrics

The `experiment_result` class in `helpers.py` aggregates `AttackResult` objects over a test set and computes:

- **ASR** (Attack Success Rate) -- fraction of correctly-classified samples that were fooled
- **Adversarial Accuracy** -- fraction of all samples still correct under attack
- **Clean Accuracy** -- fraction of all samples correct without attack

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [TabPFN](https://github.com/automl/TabPFN) (v2)
- [scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)

Install dependencies:

```bash
pip install torch tabpfn scikit-learn numpy matplotlib tqdm
```

## Usage

### Running an experiment

Each experiment can be run from the project root:

```bash
python comparison_with_classical_models/run_experiment_2b.py
```

Results are serialised to `<experiment_dir>/results/<name>.pkl` via pickle.

To use GPU acceleration, edit the `_device` variable at the bottom of each runner script (default is `"cuda"`).

### Generating figures

After running an experiment, generate thesis-quality figures:

```bash
python comparison_with_classical_models/plot_experiment_2b.py
python comparison_with_classical_models/plot_experiment_2b_aggregate.py
```

Figures are saved as PDF to `<experiment_dir>/figures/`.

## Reproducibility

- Synthetic datasets use deterministic `random_state` seeds (42, 43, ... for multi-dataset experiments)
- All data splits are stratified
- Features are standardised with `StandardScaler` (fit on train, transform on test)
- PGD attacks accept an optional `seed` parameter for reproducible random restarts
- Experiment data (including all per-sample `AttackResult` objects) is pickled for post-hoc analysis
