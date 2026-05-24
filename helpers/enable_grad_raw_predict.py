import torch
import types
import torch.nn.functional as F
from sklearn.utils.validation import check_is_fitted
from typing import Callable

def _resolve_device(device_value, fallback_tensor=None):
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
    """
    Gradient-enabled prediction that follows TabPFN's standard inference pathway
    (executor_.iter_outputs) but without torch.inference_mode().

    Returns:
        - logits (torch.Tensor) if return_logits=True
        - probabilities (torch.Tensor) if return_logits=False
    """
    check_is_fitted(self)

    if not getattr(self, "differentiable_input", False):
        raise RuntimeError(
            "differentiable_input is False. Create the classifier with "
            "differentiable_input=True and pass a torch.Tensor input."
        )

    # Ensure tensor on correct device
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

def enable_grad_raw_predict(clf, raw_predict_with_grad: Callable) -> None:
    clf._raw_predict = types.MethodType(raw_predict_with_grad, clf)