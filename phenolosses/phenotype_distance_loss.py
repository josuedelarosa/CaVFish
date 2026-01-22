from typing import List, Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from mmpose.models.heads.heatmap_heads.heatmap_head import HeatmapHead

def softargmax2d(heatmaps: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    N, K, H, W = heatmaps.shape
    logits = heatmaps.view(N, K, -1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    P = F.softmax(beta * logits, dim=-1)
    ys = torch.arange(H, device=heatmaps.device, dtype=heatmaps.dtype)
    xs = torch.arange(W, device=heatmaps.device, dtype=heatmaps.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    xx = xx.reshape(1, 1, -1); yy = yy.reshape(1, 1, -1)
    x = (P * xx).sum(-1)
    y = (P * yy).sum(-1)
    return torch.stack([x, y], dim=-1)  # (N,K,2)

def pairwise_dist(coords: torch.Tensor, i: int, j: int, eps: float = 1e-6) -> torch.Tensor:
    d = coords[:, i, :] - coords[:, j, :]
    return torch.sqrt((d * d).sum(dim=-1) + eps)

@MODELS.register_module()
class PhenotypeDistanceLoss(nn.Module):
    """MSE de distancias por pares; normaliza por SL y por grado."""
    def __init__(self,
                 pairs: List[Tuple[int, int]],
                 degree_normalize: bool = True,
                 scale_by_SL: bool = True,
                 beta: float = 15.0):
        super().__init__()
        assert len(pairs) > 0, 'Define al menos un par'
        self.pairs = pairs
        self.degree_normalize = degree_normalize
        self.scale_by_SL = scale_by_SL
        self.beta = beta

        max_k = max(max(i, j) for (i, j) in pairs) + 1 #CÁLCULO DEL GRADO POR KEYPOINT Y PESOS DE ARISTAS
        deg = [0] * max_k
        for (i, j) in pairs:
            deg[i] += 1; deg[j] += 1
        self.register_buffer('deg', torch.tensor(deg, dtype=torch.float32))
        weights = []
        for (i, j) in pairs:
            w = 1.0 / (self.deg[i] + self.deg[j]) if degree_normalize else 1.0
            weights.append(w)
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / (weights.sum() + 1e-12)
        self.register_buffer('edge_w', weights)

        self.sl_idx = None  #UBICAR EL PAR DE SL (STANDARD LENGTH)
        for idx, (i, j) in enumerate(pairs):
            if (i, j) in [(0, 1), (1, 0)]:
                self.sl_idx = idx
                break
        if self.scale_by_SL:
            assert self.sl_idx is not None, 'scale_by_SL=True requiere (0,1) en pairs'

    def _get_vis_mask(self, data_samples, K: int, device) -> torch.Tensor: #MÁSCARA DE VISIBILIDAD (N,K)
        N = len(data_samples)
        mask = torch.ones((N, K), dtype=torch.bool, device=device)
        for n, ds in enumerate(data_samples):
            gi = getattr(ds, 'gt_instances', None)
            if gi is None: continue
            vis_attr = getattr(gi, 'keypoints_visible', None)
            if vis_attr is None: continue
            vis = torch.as_tensor(vis_attr, device=device, dtype=torch.float32).view(-1)[:K] > 0.5
            mask[n, :vis.numel()] = vis
        return mask

    def _get_gt_coords_heatmap(self, data_samples, K: int, heatmap_size, device): #CONVERTIR GT DESDE COORDS DE ENTRADA A COORDS DE HEATMAP (N,K,2)
        N = len(data_samples)
        W_h, H_h = heatmap_size  # (W,H)
        gt = torch.zeros((N, K, 2), dtype=torch.float32, device=device)
        for n, ds in enumerate(data_samples):
            gi = getattr(ds, 'gt_instances', None)
            assert gi is not None, 'gt_instances no encontrado'
            kps_attr = getattr(gi, 'keypoints', None)
            assert kps_attr is not None, 'gt_instances.keypoints no encontrado'
            kps = torch.as_tensor(kps_attr, device=device, dtype=torch.float32)
            if kps.dim() == 3:  # (num_inst,K,2) -> usa la primera
                kps = kps[0]
            meta = getattr(ds, 'metainfo', {})
            input_size = meta.get('input_size', None)  # (W_in,H_in)
            assert input_size is not None, 'metainfo.input_size requerido'
            W_in, H_in = input_size
            sx, sy = W_h / float(W_in), H_h / float(H_in)
            kps_hm = kps.clone()
            kps_hm[:, 0] *= sx
            kps_hm[:, 1] *= sy
            gt[n, :min(K, kps_hm.shape[0]), :] = kps_hm[:K, :]
        return gt

#CÁLCULO DE LA PÉRDIDA
    def forward(self, pred_heatmaps: torch.Tensor, data_samples: list, heatmap_size=None) -> torch.Tensor:
        device = pred_heatmaps.device
        N, K, H, W = pred_heatmaps.shape
        pred_xy = softargmax2d(pred_heatmaps, beta=self.beta)
        if heatmap_size is None:
            meta0 = getattr(data_samples[0], 'metainfo', {})
            heatmap_size = meta0.get('heatmap_size', (W, H))  # (W,H)
        gt_xy = self._get_gt_coords_heatmap(data_samples, K, heatmap_size, device)
        vis_k = self._get_vis_mask(data_samples, K, device)  # (N,K)
        pair_masks = torch.stack([(vis_k[:, i] & vis_k[:, j]) for (i, j) in self.pairs], dim=1).float()
        pred_d = torch.stack([pairwise_dist(pred_xy, i, j) for (i, j) in self.pairs], dim=1)
        gt_d   = torch.stack([pairwise_dist(gt_xy,   i, j) for (i, j) in self.pairs], dim=1)
        if self.scale_by_SL:
            sl = self.sl_idx
            pred_SL = pred_d[:, sl:sl+1] + 1e-6
            gt_SL   = gt_d[:, sl:sl+1]   + 1e-6
            pred_d = pred_d / pred_SL
            gt_d   = gt_d   / gt_SL
        Wm = self.edge_w.to(device).unsqueeze(0).expand_as(pred_d)
        residual = (pred_d - gt_d) * pair_masks
        loss_mat = (residual ** 2) * Wm
        valid_w = (Wm * pair_masks).sum(dim=1) + 1e-12
        return (loss_mat.sum(dim=1) / valid_w).mean()