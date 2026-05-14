"""Fast Gradient Sign Method (FGSM) attacks (Goodfellow et al., 2015).

Provides single-step L-infinity adversarial perturbations for three model
families: TabPFN, standard PyTorch neural networks, and scikit-learn linear
classifiers (LogisticRegression, LinearSVC, SVC with linear kernel).
"""
from __future__ import annotations

import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from attacks_common import AttackResult, _predict_logits, _to_numpy_1d, _to_python_label


def fgsm_attack(
    clf,
    x_nat: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: Union[float, torch.Tensor] = 0.25,
    targeted: bool = False,
    loss_reduction: str = "mean",
) -> AttackResult:
    """FGSM attack for TabPFN classifiers.

    Computes the adversarial perturbation x_adv = x + eps * sign(grad_x L)
    using the cross-entropy loss. Requires the classifier to expose a
    differentiable ``_raw_predict`` method (see ``enable_grad_raw_predict``).

    Args:
        clf: TabPFN classifier with a differentiable _raw_predict method.
        x_nat: Clean input tensor, shape [d] or [1, d].
        y: True class label(s), scalar or shape [N].
        eps: L-infinity perturbation budget (scalar or per-feature tensor).
        targeted: If True, minimise loss toward y (targeted attack).
        loss_reduction: Reduction for cross-entropy ('mean' or 'sum').
    """
    if not torch.is_tensor(x_nat):
        raise TypeError("x_nat must be a torch.Tensor.")
    if not torch.is_tensor(y):
        raise TypeError("y must be a torch.Tensor of class indices.")

    if x_nat.ndim == 1:
        x_nat_b = x_nat.unsqueeze(0)
    else:
        x_nat_b = x_nat

    if y.ndim == 0:
        y_b = y.unsqueeze(0)
    else:
        y_b = y

    if y_b.ndim != 1:
        raise ValueError(f"y must be a scalar or 1D tensor of shape [N]; got shape {tuple(y.shape)}.")

    if y_b.shape[0] != x_nat_b.shape[0]:
        raise ValueError(
            f"Batch size mismatch: x_nat has N={x_nat_b.shape[0]} but y has N={y_b.shape[0]}."
        )

    y_b = y_b.to(device=x_nat_b.device, dtype=torch.long)

    logits_nat = _predict_logits(clf, x_nat_b)
    if logits_nat.ndim != 2:
        raise ValueError(f"Expected logits [N,C]; got {tuple(logits_nat.shape)}.")
    pred_nat = int(logits_nat.argmax(dim=1)[0].item())

    x = x_nat_b.detach().clone()
    x.requires_grad_(True)

    logits = clf._raw_predict(x, return_logits=True)
    loss = F.cross_entropy(logits, y_b, reduction=loss_reduction)
    grad = torch.autograd.grad(loss, x, only_inputs=True)[0]
    if grad is None:
        raise RuntimeError("Failed to compute gradient wrt input (grad is None).")

    step = -1.0 if targeted else 1.0

    if torch.is_tensor(eps):
        eps_t = eps.to(device=x.device, dtype=x.dtype)
    else:
        eps_t = torch.tensor(eps, device=x.device, dtype=x.dtype)

    x_adv = x + step * eps_t * grad.sign()
    x_adv = x_adv.detach()

    logits_adv = _predict_logits(clf, x_adv)
    loss_adv = float(F.cross_entropy(logits_adv, y_b, reduction="mean").item())
    pred_adv = int(logits_adv.argmax(dim=1)[0].item())

    return AttackResult(
        x_adv=x_adv,
        best_loss=loss_adv,
        pred_adv=pred_adv,
        pred_nat=pred_nat,
        y_true=int(y_b[0].item()),
    )


def fgsm_attack_nn(
    model: torch.nn.Module,
    x_nat: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps: float = 0.25,
) -> AttackResult:
    """FGSM attack for standard PyTorch neural networks.

    Args:
        model: Any ``torch.nn.Module`` producing logits of shape [N, C].
        x_nat: Clean input tensor, shape [d] or [1, d].
        y_true: True class label, scalar or shape [1].
        eps: L-infinity perturbation budget.
    """
    device = next(model.parameters()).device

    if x_nat.ndim == 1:
        x_nat = x_nat.unsqueeze(0)
    if y_true.ndim == 0:
        y_true = y_true.unsqueeze(0)

    x_nat = x_nat.to(device)
    y_true = y_true.to(device, dtype=torch.long)

    model.eval()

    with torch.no_grad():
        logits_nat = model(x_nat)
        pred_nat = int(logits_nat.argmax(dim=1).item())

    x = x_nat.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y_true)
    grad = torch.autograd.grad(loss, x, only_inputs=True)[0]

    x_adv = x + eps * grad.sign()
    x_adv = x_adv.detach()

    with torch.no_grad():
        logits_adv = model(x_adv)
        loss_adv = float(F.cross_entropy(logits_adv, y_true).item())
        pred_adv = int(logits_adv.argmax(dim=1).item())

    return AttackResult(
        x_adv=x_adv,
        best_loss=loss_adv,
        pred_adv=pred_adv,
        pred_nat=pred_nat,
        y_true=int(y_true[0].item()),
    )


def fgsm_attack_sklearn(
    clf,
    x_nat,
    y_true,
    eps: float = 0.1,
) -> AttackResult:
    """FGSM attack for scikit-learn linear binary classifiers.

    Computes the closed-form input gradient from the model's weight vector
    ``coef_``. Supports LogisticRegression (cross-entropy gradient),
    LinearSVC, and SVC with a linear kernel (hinge-loss gradient).

    Args:
        clf: Fitted sklearn classifier with ``coef_`` and ``classes_``.
        x_nat: Clean input (numpy array or torch tensor).
        y_true: True class label.
        eps: L-infinity perturbation budget.
    """
    x = _to_numpy_1d(x_nat)
    y = _to_python_label(y_true)

    if not hasattr(clf, "classes_") or len(clf.classes_) != 2:
        raise ValueError("This FGSM implementation supports only binary classifiers with clf.classes_ of length 2.")

    classes = clf.classes_
    if y not in classes:
        raise ValueError(f"y_true={y} not in clf.classes_={classes}.")

    pred_nat = clf.predict(x.reshape(1, -1))[0]

    if not hasattr(clf, "coef_"):
        raise ValueError("Classifier has no coef_. This implementation requires a linear model.")

    w = np.asarray(clf.coef_).reshape(-1)
    b = float(np.asarray(clf.intercept_).reshape(-1)[0]) if hasattr(clf, "intercept_") else 0.0

    y01 = 1 if y == classes[1] else 0
    ypm = 1.0 if y01 == 1 else -1.0

    if isinstance(clf, LogisticRegression):
        z = float(np.dot(w, x) + b)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (p - float(y01)) * w  # dL/dx for binary cross-entropy

    elif isinstance(clf, LinearSVC) or (isinstance(clf, SVC) and getattr(clf, "kernel", None) == "linear"):
        grad = -ypm * w

    else:
        raise ValueError(f"Unsupported classifier type: {type(clf)}. Use LogisticRegression, LinearSVC, or SVC(kernel='linear').")

    x_adv_np = x + eps * np.sign(grad)

    pred_adv = clf.predict(x_adv_np.reshape(1, -1))[0]

    if isinstance(clf, LogisticRegression):
        z_adv = float(np.dot(w, x_adv_np) + b)
        p_adv = 1.0 / (1.0 + np.exp(-z_adv))
        p_adv = np.clip(p_adv, 1e-12, 1.0 - 1e-12)
        loss_adv = -(y01 * np.log(p_adv) + (1 - y01) * np.log(1.0 - p_adv))
    else:
        s_adv = float(np.dot(w, x_adv_np) + b)
        margin_adv = ypm * s_adv
        loss_adv = max(0.0, 1.0 - margin_adv)

    x_adv_t = torch.tensor(x_adv_np, dtype=torch.float32).unsqueeze(0)

    return AttackResult(
        x_adv=x_adv_t,
        best_loss=float(loss_adv),
        pred_adv=int(pred_adv),
        pred_nat=int(pred_nat),
        y_true=int(y),
    )