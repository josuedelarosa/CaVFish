import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.registry import MODELS


@MODELS.register_module()
class KeypointLogMSELoss(nn.Module):
    """log(MSE) loss for heatmaps.

    Es igual a KeypointMSELoss, pero después de calcular el MSE
    aplicamos log(mse + eps).

    Args:
        use_target_weight (bool): Si True, aplica pesos por keypoint.
        skip_empty_channel (bool): Si True, ignora canales sin keypoint visible.
        eps (float): pequeño valor para estabilidad numérica en log.
        loss_weight (float): Peso global de esta pérdida.
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 eps: float = 1e-6,
                 loss_weight: float = 1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Tensor = None,
                mask: Tensor = None) -> Tensor:
        """Calcula log(MSE + eps).

        Parámetros:
            output: [B, K, H, W] heatmaps predichos
            target: [B, K, H, W] heatmaps GT
            target_weights: [B, K] o [B, K, H, W] (opcional)
            mask: [B, K, H, W] o [B, 1, H, W] (opcional)
        """

        # 1. Reproducimos exactamente la misma lógica de máscara
        _mask = self._get_mask(target, target_weights, mask)

        if _mask is None:
            # mse global estándar
            mse_val = F.mse_loss(output, target, reduction='mean')
        else:
            # mse pixel a pixel
            _loss = F.mse_loss(output, target, reduction='none')  # [B,K,H,W]
            # aplicamos máscara y luego promediamos
            mse_val = (_loss * _mask).mean()

        # 2. aplicamos log(mse + eps)
        log_mse_val = torch.log(mse_val + self.eps)

        # 3. escalamos por loss_weight
        return log_mse_val * self.loss_weight

    def _get_mask(self, target: Tensor, target_weights: Tensor,
                  mask: Tensor) -> Tensor:
        """Misma función que la original KeypointMSELoss."""
        # Esta parte es casi copia literal de tu clase original,
        # para que el comportamiento sea idéntico excepto por el log final.

        # Given spatial mask
        if mask is not None:
            assert (mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1
                for d_m, d_t in zip(mask.shape, target.shape))), (
                    f'mask and target have mismatched shapes {mask.shape} v.s.'
                    f'{target.shape}')

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            assert (target_weights.ndim in (2, 4)
                    and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(
                target_weights.shape + (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask
