
from typing import List, Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from mmpose.models.heads.heatmap_heads.heatmap_head import HeatmapHead

@MODELS.register_module() 
class HeatmapHeadPheno(HeatmapHead):
    """HeatmapHead + término de pérdida fenotípica adicional."""
    def __init__(self, loss_pheno: dict, alpha_pheno: float = 1e-2, **kwargs):
        super().__init__(**kwargs)
        self.loss_pheno = MODELS.build(loss_pheno)
        self.alpha_pheno = float(alpha_pheno)

    # Reuse predictions that the base class already computed
    def loss_by_feat(self, preds: Tuple[Tensor], data_samples: Optional[List] = None):
        # 1) compute the base heatmap losses using the parent implementation
        losses = super().loss_by_feat(preds, data_samples)

        # 2) add phenotype loss using the same preds (no extra forward)
        #    Most heatmap heads return a single tensor; guard tuple/list for safety
        if isinstance(preds, (list, tuple)):
            pred = preds[0]
        else:
            pred = preds

        meta0 = getattr(data_samples[0], 'metainfo', {}) if data_samples else {}
        hm_size = meta0.get('heatmap_size', None)  # (W, H)

        loss_ph = self.loss_pheno(
            pred_heatmaps=pred,
            data_samples=data_samples,
            heatmap_size=hm_size
        )
        losses['loss_pheno'] = self.alpha_pheno * loss_ph
        return losses