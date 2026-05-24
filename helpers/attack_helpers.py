import torch
import numpy as np

@torch.no_grad()
def _predict_logits(clf, x: torch.Tensor) -> torch.Tensor:
    """Helper to get logits without tracking gradients."""
    return clf._raw_predict(x, return_logits=True)


def _to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x


def _to_python_label(y):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().item()
    return y
