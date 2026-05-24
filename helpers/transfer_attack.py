import torch
from helpers.AttackResult import AttackResult
from helpers.attack_helpers import _to_python_label
from helpers.model_helpers import _predict_target


def transfer_attack(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 0.25,
    source_clf,
    source_attack_fn,
    source_attack_args: dict = None,
    target_clf,
    target_name: str,
) -> AttackResult:
    if source_attack_args is None:
        source_attack_args = {}

    y_true = _to_python_label(y)
    target_tuple = (target_name, target_clf)
    pred_nat = _predict_target(target_tuple, x)

    if eps == 0.0:
        x_adv_nat = x.detach().clone() if isinstance(x, torch.Tensor) else x
        return AttackResult(
            x_adv=x_adv_nat,
            best_loss=0.0,
            pred_adv=pred_nat,
            pred_nat=pred_nat,
            y_true=y_true,
        )

    source_result = source_attack_fn(source_clf, x, y, eps=eps, **source_attack_args)
    x_adv = source_result.x_adv
    pred_adv = _predict_target(target_tuple, x_adv)

    return AttackResult(
        x_adv=x_adv,
        best_loss=float(source_result.best_loss),
        pred_adv=pred_adv,
        pred_nat=pred_nat,
        y_true=y_true,
    )
