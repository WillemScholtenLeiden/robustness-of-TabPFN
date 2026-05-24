from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class AttackResult:
    x_adv: torch.Tensor
    best_loss: float
    pred_adv: int
    pred_nat: int
    y_true: int