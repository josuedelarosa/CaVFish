from __future__ import annotations
import torch
import torch.nn as nn
from mmpose.registry import MODELS

@MODELS.register_module()
class CoordSmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) for coordinates.
    pred, target: [B, K, 2]
    beta: transition point (0 → pure L1).
    reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.beta = float(beta)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs):
        diff = (pred - target).abs()
        if self.beta > 0:
            loss = torch.where(
                diff < self.beta,
                0.5 * diff * diff / self.beta,
                diff - 0.5 * self.beta
            )
        else:
            loss = diff  # beta==0 => L1

        if self.reduction == 'none':
            return loss         # [B,K,2]
        if self.reduction == 'sum':
            return loss.sum()
        return loss.mean()
