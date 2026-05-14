"""Binary-search routines for finding minimum adversarial epsilon."""
from __future__ import annotations

from typing import Callable, Optional

import torch

from attacks_common import _predict_logits, _to_python_label
from fgsm_attack import fgsm_attack
from pgd_attack import pgd_linf_restarts


def find_upper_bound(
    clf,
    x: torch.Tensor,
    y: torch.Tensor,
    eps_lo: float = 0.0,
    eps_hi: float = 4.0,
    tol: float = 1e-3,
) -> float | None:
    """Binary-search for the smallest FGSM epsilon causing misclassification.

    Returns:
        The threshold epsilon, 0.0 if already misclassified, or None if
        the sample is robust beyond ``eps_hi``.
    """
    result_nat = fgsm_attack(clf, x, y, eps=0.0)
    if result_nat.pred_nat != result_nat.y_true:
        return 0.0

    result_hi = fgsm_attack(clf, x, y, eps=eps_hi)
    if result_hi.pred_adv == result_hi.y_true:
        return None

    while (eps_hi - eps_lo) > tol:
        eps_mid = (eps_lo + eps_hi) / 2.0
        result = fgsm_attack(clf, x, y, eps=eps_mid)
        flipped = result.pred_adv != result.y_true
        del result
        if flipped:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid
    return eps_hi


def find_upper_bound_pgd(
    clf,
    x: torch.Tensor,
    y: torch.Tensor,
    eps_lo: float = 0.0,
    eps_hi: float = 4.0,
    tol: float = 1e-3,
    pgd_steps: int = 5,
    pgd_restarts: int = 5,
    seed: Optional[int] = None,
) -> float | None:
    """Binary-search for the smallest PGD epsilon causing misclassification.

    Returns:
        The threshold epsilon, 0.0 if already misclassified, or None if
        the sample is robust beyond ``eps_hi``.
    """
    result_nat = pgd_linf_restarts(clf, x, y, eps=0.0, steps=pgd_steps, restarts=pgd_restarts, seed=seed)
    if result_nat.pred_nat != result_nat.y_true:
        return 0.0

    result_hi = pgd_linf_restarts(clf, x, y, eps=eps_hi, steps=pgd_steps, restarts=pgd_restarts, seed=seed)
    if result_hi.pred_adv == result_hi.y_true:
        return None

    while (eps_hi - eps_lo) > tol:
        eps_mid = (eps_lo + eps_hi) / 2.0
        result = pgd_linf_restarts(clf, x, y, eps=eps_mid, steps=pgd_steps, restarts=pgd_restarts, seed=seed)
        flipped = result.pred_adv != result.y_true
        del result
        if flipped:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid
    return eps_hi


def find_upper_bound_sklearn(
    clf,
    x: torch.Tensor,
    y: torch.Tensor,
    attack_fn: Callable,
    eps_lo: float = 0.0,
    eps_hi: float = 4.0,
    tol: float = 1e-3,
    **attack_kwargs,
) -> float | None:
    """Binary-search for the smallest epsilon causing misclassification (sklearn).

    Returns:
        The threshold epsilon, 0.0 if already misclassified, or None if
        the sample is robust beyond ``eps_hi``.
    """
    result_nat = attack_fn(clf, x, y, eps=0.0, **attack_kwargs)
    if result_nat.pred_nat != result_nat.y_true:
        return 0.0

    result_hi = attack_fn(clf, x, y, eps=eps_hi, **attack_kwargs)
    if result_hi.pred_adv == result_hi.y_true:
        return None

    while (eps_hi - eps_lo) > tol:
        eps_mid = (eps_lo + eps_hi) / 2.0
        result = attack_fn(clf, x, y, eps=eps_mid, **attack_kwargs)
        if result.pred_adv != result.y_true:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid
    return eps_hi


def find_upper_bound_transfer(
    source_clf,
    source_attack_fn: Callable,
    target_clf,
    x: torch.Tensor,
    y: torch.Tensor,
    eps_lo: float = 0.0,
    eps_hi: float = 4.0,
    tol: float = 1e-3,
    device: str = "cpu",
    **attack_kwargs,
) -> float | None:
    """Binary-search for the smallest epsilon at which a source-model adversarial example fools the target.

    Generates adversarial examples on ``source_clf`` and evaluates whether
    they transfer to ``target_clf`` (a TabPFN model).

    Returns:
        The threshold epsilon, 0.0 if already misclassified, or None if
        the sample is robust beyond ``eps_hi``.
    """
    x_b = x.unsqueeze(0) if x.ndim == 1 else x
    x_b = x_b.to(device)
    y_t = torch.tensor([_to_python_label(y)], dtype=torch.long, device=device)
    logits_nat = _predict_logits(target_clf, x_b)
    pred_nat = int(logits_nat.argmax(dim=1)[0].item())
    if pred_nat != int(y_t[0].item()):
        return 0.0

    result_hi = source_attack_fn(source_clf, x, y, eps=eps_hi, **attack_kwargs)
    x_adv_hi = result_hi.x_adv.to(device)
    if x_adv_hi.ndim == 1:
        x_adv_hi = x_adv_hi.unsqueeze(0)
    logits_adv = _predict_logits(target_clf, x_adv_hi)
    pred_adv = int(logits_adv.argmax(dim=1)[0].item())
    if pred_adv == int(y_t[0].item()):
        return None

    y_true_val = int(y_t[0].item())
    while (eps_hi - eps_lo) > tol:
        eps_mid = (eps_lo + eps_hi) / 2.0
        result = source_attack_fn(source_clf, x, y, eps=eps_mid, **attack_kwargs)
        x_adv = result.x_adv.to(device)
        if x_adv.ndim == 1:
            x_adv = x_adv.unsqueeze(0)
        logits_adv = _predict_logits(target_clf, x_adv)
        pred_adv = int(logits_adv.argmax(dim=1)[0].item())
        del result, x_adv, logits_adv
        if pred_adv != y_true_val:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid
    return eps_hi
