"""Projected Gradient Descent (PGD) attacks (Madry et al., 2018).

Provides multi-step, multi-restart L-infinity adversarial perturbations for
three model families: TabPFN, standard PyTorch neural networks, and
scikit-learn linear classifiers (LogisticRegression, LinearSVC, SVC with
linear kernel).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from tqdm import tqdm

from attacks_common import AttackResult, _predict_logits, _to_numpy_1d, _to_python_label
from raw_predict_with_grad import enable_grad_raw_predict


def pgd_linf_restarts(
    clf,
    x_nat: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps: float = 0.25,
    steps: int = 40,
    restarts: int = 20,
    seed: Optional[int] = None,
) -> AttackResult:
    """PGD L-inf attack with random restarts for TabPFN classifiers.

    Each restart initialises a random point inside the L-inf ball of radius
    ``eps`` around ``x_nat``, then runs ``steps`` of signed gradient ascent
    on the cross-entropy loss. The best adversarial example (highest loss)
    across all restarts is returned.

    Args:
        clf: TabPFN classifier with a differentiable _raw_predict method.
        x_nat: Clean input tensor, shape [d] or [1, d].
        y_true: True class label, scalar or shape [1].
        eps: L-infinity perturbation budget.
        steps: Gradient-ascent iterations per restart.
        restarts: Number of random restarts.
        seed: Optional RNG seed for reproducibility.
    """
    alpha = eps / steps * 2

    if x_nat.ndim == 1:
        x_nat = x_nat.unsqueeze(0)
    if y_true.ndim == 0:
        y_true = y_true.unsqueeze(0)
    if y_true.ndim == 1 and y_true.shape[0] != x_nat.shape[0]:
        y_true = y_true[:1]

    device = x_nat.device
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    logits_nat = _predict_logits(clf, x_nat)
    pred_nat = int(logits_nat.argmax(dim=1).item())
    best_loss = float(F.cross_entropy(logits_nat, y_true).item())
    best_x = x_nat.clone().detach()
    best_pred = pred_nat

    for _ in tqdm(range(restarts), desc="Restarts", leave=False):
        if g is None:
            noise = torch.empty_like(x_nat).uniform_(-eps, eps)
        else:
            noise = torch.empty_like(x_nat).uniform_(-eps, eps, generator=g)

        x = (x_nat + noise).detach()

        for _ in range(steps):
            x.requires_grad_(True)

            logits = clf._raw_predict(x, return_logits=True)
            loss = F.cross_entropy(logits, y_true)

            grad = torch.autograd.grad(loss, x, only_inputs=True)[0]

            with torch.no_grad():
                x = x + alpha * grad.sign()
                x = torch.max(torch.min(x, x_nat + eps), x_nat - eps)
                x = x.detach()

        logits = _predict_logits(clf, x)
        loss_val = float(F.cross_entropy(logits, y_true).item())
        pred = int(logits.argmax(dim=1).item())

        if loss_val > best_loss:
            best_loss = loss_val
            best_x = x.clone().detach()
            best_pred = pred

    return AttackResult(
        x_adv=best_x,
        best_loss=best_loss,
        pred_adv=best_pred,
        pred_nat=pred_nat,
        y_true=int(y_true[0].item()),
    )

def pgd_attack_nn(
    model: torch.nn.Module,
    x_nat: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps: float = 0.25,
    steps: int = 40,
    restarts: int = 10,
) -> AttackResult:
    """PGD L-inf attack with random restarts for standard PyTorch networks.

    Same algorithm as ``pgd_linf_restarts`` but operates on any
    ``torch.nn.Module`` that outputs logits of shape [N, C].

    Args:
        model: Any ``torch.nn.Module`` producing logits of shape [N, C].
        x_nat: Clean input tensor, shape [d] or [1, d].
        y_true: True class label, scalar or shape [1].
        eps: L-infinity perturbation budget.
        steps: Gradient-ascent iterations per restart.
        restarts: Number of random restarts.
    """
    alpha = eps / steps * 2
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
        best_loss = float(F.cross_entropy(logits_nat, y_true).item())

    best_x = x_nat.clone().detach()
    best_pred = pred_nat

    for _ in range(restarts):
        noise = torch.empty_like(x_nat).uniform_(-eps, eps)
        x = (x_nat + noise).detach()

        for _ in range(steps):
            x.requires_grad_(True)
            logits = model(x)
            loss = F.cross_entropy(logits, y_true)
            grad = torch.autograd.grad(loss, x, only_inputs=True)[0]

            with torch.no_grad():
                x = x + alpha * grad.sign()
                x = torch.max(torch.min(x, x_nat + eps), x_nat - eps)
                x = x.detach()

        with torch.no_grad():
            logits = model(x)
            loss_val = float(F.cross_entropy(logits, y_true).item())
            pred = int(logits.argmax(dim=1).item())

        if loss_val > best_loss:
            best_loss = loss_val
            best_x = x.clone().detach()
            best_pred = pred

    return AttackResult(
        x_adv=best_x,
        best_loss=best_loss,
        pred_adv=best_pred,
        pred_nat=pred_nat,
        y_true=int(y_true[0].item()),
    )



def pgd_attack_sklearn(
        clf,
        x_nat,
        y_true,
        eps: float = 0.25,
        *,
        steps: int = 40,
        restarts: int = 10,
        seed: Optional[int] = None,
) -> AttackResult:
    """PGD L-inf attack with random restarts for scikit-learn linear classifiers.

    Uses the closed-form gradient from the model's weight vector ``coef_``.
    Supports LogisticRegression (cross-entropy), LinearSVC and SVC with a
    linear kernel (hinge loss).

    Args:
        clf: Fitted sklearn classifier with ``coef_`` and ``classes_``.
        x_nat: Clean input (numpy array or torch tensor).
        y_true: True class label.
        eps: L-infinity perturbation budget.
        steps: Gradient-ascent iterations per restart.
        restarts: Number of random restarts.
        seed: Optional RNG seed for reproducibility.
    """
    alpha = eps / steps * 2
    x = _to_numpy_1d(x_nat)
    y = _to_python_label(y_true)

    classes = clf.classes_
    pred_nat = clf.predict(x.reshape(1, -1))[0]

    w = np.asarray(clf.coef_).reshape(-1)
    b = float(np.asarray(clf.intercept_).reshape(-1)[0]) if hasattr(clf, "intercept_") else 0.0

    y01 = 1 if y == classes[1] else 0
    ypm = 1.0 if y01 == 1 else -1.0

    rng = np.random.default_rng(seed)

    def compute_loss_grad(x_curr):
        """Compute loss and input gradient for the current point."""
        if isinstance(clf, LogisticRegression):
            z = float(np.dot(w, x_curr) + b)
            # Numerically stable sigmoid: avoid overflow in exp()
            if z >= 0:
                exp_neg_z = np.exp(-z)
                p = 1.0 / (1.0 + exp_neg_z)
            else:
                exp_z = np.exp(z)
                p = exp_z / (1.0 + exp_z)

            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            loss = -(y01 * np.log(p) + (1 - y01) * np.log(1.0 - p))

            grad_loss = (p - float(y01)) * w

        elif isinstance(clf, LinearSVC) or (isinstance(clf, SVC) and getattr(clf, "kernel", None) == "linear"):
            s = float(np.dot(w, x_curr) + b)
            margin = ypm * s

            loss = 1.0 - margin
            grad_loss = -ypm * w
        else:
            raise ValueError(f"Unsupported classifier type: {type(clf)}.")

        return loss, grad_loss

    best_loss, _ = compute_loss_grad(x)
    best_x = x.copy()
    best_pred = pred_nat

    for _ in range(restarts):
        noise = rng.uniform(-eps, eps, size=x.shape)
        x_curr = x + noise

        for _ in range(steps):
            loss_val, grad = compute_loss_grad(x_curr)

            x_curr = x_curr + alpha * np.sign(grad)
            x_curr = np.maximum(np.minimum(x_curr, x + eps), x - eps)

        loss_val, _ = compute_loss_grad(x_curr)
        pred_curr = clf.predict(x_curr.reshape(1, -1))[0]

        if loss_val > best_loss:
            best_loss = loss_val
            best_x = x_curr.copy()
            best_pred = pred_curr

    x_adv_t = torch.tensor(best_x, dtype=torch.float32).unsqueeze(0)

    return AttackResult(
        x_adv=x_adv_t,
        best_loss=float(best_loss),
        pred_adv=int(best_pred),
        pred_nat=int(pred_nat),
        y_true=int(y),
    )