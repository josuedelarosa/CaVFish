# vitpose_head_pheno.py
from typing import List, Tuple, Optional, Union
import torch
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.models.heads.heatmap_heads.heatmap_head import HeatmapHead

@MODELS.register_module(name='MinPhenoHead')
class MinPhenoHead(HeatmapHead):
    """HeatmapHead + término de pérdida fenotípica adicional."""
    def __init__(self, loss_pheno: dict, alpha_pheno: float = 1e-2, **kwargs):
        super().__init__(**kwargs)
        self.loss_pheno = MODELS.build(loss_pheno)
        self.alpha_pheno = float(alpha_pheno)

    def loss_by_feat(self, preds: Union[Tensor, Tuple[Tensor, ...], List[Tensor]],
                     data_samples: Optional[List] = None):
        # 1) pérdidas base del heatmap (MSE/whatever)
        losses = super().loss_by_feat(preds, data_samples)

        # 2) pérdida fenotípica
        stages = preds if isinstance(preds, (list, tuple)) else [preds]
        loss_ph_total = None
        # Pasamos el heatmap_size por metainfo (si existe) para ser explícitos
        meta0 = getattr(data_samples[0], 'metainfo', {}) if data_samples else {}
        hm_size = meta0.get('heatmap_size', None)  # (W, H)

        for ph in stages:
            lp = self.loss_pheno(pred_heatmaps=ph,
                                 data_samples=data_samples,
                                 heatmap_size=hm_size)
            loss_ph_total = lp if loss_ph_total is None else (loss_ph_total + lp)

        losses['loss_pheno'] = self.alpha_pheno * loss_ph_total
        return losses
