"""Shared data structures and helpers for adversarial attack modules."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AttackResult:
    """Result of a single-sample adversarial attack.

    Attributes:
        x_adv: Perturbed input tensor, shape [1, d].
        best_loss: Loss achieved by the adversarial example.
        pred_adv: Predicted class on the perturbed input.
        pred_nat: Predicted class on the clean input.
        y_true: Ground-truth label.
    """
    x_adv: torch.Tensor
    best_loss: float
    pred_adv: int
    pred_nat: int
    y_true: int


@torch.no_grad()
def _predict_logits(clf, x: torch.Tensor) -> torch.Tensor:
    """Return raw logits from a TabPFN classifier without gradient tracking."""
    return clf._raw_predict(x, return_logits=True)

def _to_numpy_1d(x):
    """Convert a tensor or array to a flat float64 numpy vector."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x


def _to_python_label(y):
    """Extract a plain Python scalar from a tensor or native label."""
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().item()
    return y
