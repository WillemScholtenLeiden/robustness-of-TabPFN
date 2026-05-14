"""Gradient-enabled inference for TabPFN classifiers.

Provides ``raw_predict_with_grad``, a drop-in replacement for
TabPFN's ``_raw_predict`` that keeps the autograd graph alive,
enabling gradient-based adversarial attacks on TabPFN.
"""
import types

import torch
import torch.nn.functional as F
from sklearn.utils.validation import check_is_fitted


def _resolve_device(device_value, fallback_tensor=None):
    """Resolve a device specification to a ``torch.device`` object."""
    if device_value is None or device_value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if isinstance(device_value, torch.device):
        return device_value

    return torch.device(device_value)

def raw_predict_with_grad(self, X, *, return_logits: bool = True):
    """Gradient-enabled prediction via TabPFN's executor without inference_mode.

    Args:
        X: Input tensor or array-like.
        return_logits: If True return raw logits, otherwise softmax probabilities.
    """
    check_is_fitted(self)

    if not getattr(self, "differentiable_input", False):
        raise RuntimeError(
            "differentiable_input is False. Create the classifier with "
            "differentiable_input=True and pass a torch.Tensor input."
        )

    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)

    device_attr = getattr(self, "device_", None) or getattr(self, "device", None)
    device = _resolve_device(device_attr)
    X = X.to(device)

    outputs = []
    autocast_enabled = bool(getattr(self, "use_autocast_", False))

    with torch.enable_grad():
        for out, _cfg in self.executor_.iter_outputs(
            X,
            autocast=autocast_enabled,
        ):

            if isinstance(out, (tuple, list)):
                out = out[0]
            outputs.append(out)

    if len(outputs) == 0:
        raise RuntimeError("No outputs produced by executor_.iter_outputs(...).")

    logits = torch.stack(outputs, dim=0).mean(dim=0) if len(outputs) > 1 else outputs[0]

    if return_logits:
        return logits
    return F.softmax(logits, dim=-1)


def enable_grad_raw_predict(clf, raw_predict_fn) -> None:
    """Monkey-patch a TabPFN classifier so _raw_predict supports autograd.

    Replaces the default (no-grad) ``_raw_predict`` with a version that
    keeps the computation graph, allowing gradient-based attacks.
    """
    clf._raw_predict = types.MethodType(raw_predict_fn, clf)
