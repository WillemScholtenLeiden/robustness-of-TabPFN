import torch
import numpy as np
from helpers.enable_grad_raw_predict import enable_grad_raw_predict, raw_predict_with_grad

def _predict_target(model: tuple, x_adv: torch.Tensor) -> int:
    name, clf = model[0], model[1]
    if x_adv.ndim == 2:
        x_adv = x_adv.squeeze(0)
    if name[:6].lower() == 'tabpfn':
        x_in = x_adv.unsqueeze(0) if x_adv.ndim == 1 else x_adv
        with torch.no_grad():
            logits = clf._raw_predict(x_in, return_logits=True)
        return int(logits.argmax(dim=1).item())
    else:
        x_np = x_adv.detach().cpu().numpy().reshape(1, -1)
        return int(clf.predict(x_np)[0])

def _train_model(model: tuple, X_train: torch.Tensor, y_train: torch.Tensor,
                X_train_np: np.ndarray, y_train_np: np.ndarray) -> None:
    if model[0][:6].lower() == 'tabpfn':
        model[1].fit(X_train, y_train)
        enable_grad_raw_predict(model[1], raw_predict_with_grad)
    else:
        model[1].fit(X_train_np, y_train_np)


def _train_models(models: list, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()

    for model in models:
        _train_model(model, X_train, y_train, X_train_np, y_train_np)