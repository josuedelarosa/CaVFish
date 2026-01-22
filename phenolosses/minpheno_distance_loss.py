# phenotype_distance_loss.py
from typing import List, Tuple, Optional, Union, Dict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS

# ------- small cache for coordinate grid to speed softargmax -------
_grid_cache: Dict[Tuple[int, int, torch.device, torch.dtype], Tuple[Tensor, Tensor]] = {}

def _get_grid(H: int, W: int, device, dtype):
    key = (H, W, device, dtype)
    g = _grid_cache.get(key)
    if g is None:
        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        _grid_cache[key] = (xx.reshape(1, 1, -1), yy.reshape(1, 1, -1))
        g = _grid_cache[key]
    return g

def softargmax2d(heatmaps: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Returns (N, K, 2) coordinates in heatmap space."""
    N, K, H, W = heatmaps.shape
    logits = heatmaps.reshape(N, K, -1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    P = F.softmax(beta * logits.float(), dim=-1).to(heatmaps.dtype)
    xx, yy = _get_grid(H, W, heatmaps.device, heatmaps.dtype)
    x = (P * xx).sum(-1)
    y = (P * yy).sum(-1)
    return torch.stack([x, y], dim=-1)

def pairwise_dist(coords: torch.Tensor, i: int, j: int, eps: float = 1e-6) -> torch.Tensor:
    d = coords[:, i, :] - coords[:, j, :]
    return torch.sqrt((d * d).sum(dim=-1) + eps)

@MODELS.register_module()
class MinPhenotypeDistanceLoss(nn.Module):
    """
    Pairwise distance MSE with (optional) degree weighting and scale normalization.

    Normalization options:
      - "sl":      divide all distances by the SL pair distance (GT)  -> legacy
      - "min_gt":  divide by the smallest GT distance among min_pairs (visibility-aware)
      - "min_pred":divide by the smallest *pred* distance among min_pairs (visibility-aware)
      - "none":    no normalization

    Extra safety knobs:
      - detach_scale (bool): remove gradients through the scale (recommended True)
      - percentile (float|None): use e.g. 10.0 instead of hard min to avoid outliers
      - clamp_min / clamp_max: clamp the scale before dividing (prevents blow-ups)
    """
    def __init__(self,
                 pairs: List[Tuple[int, int]],
                 degree_normalize: bool = True,
                 normalization: str = "min_pred",          # "sl" | "min_gt" | "min_pred" | "none"
                 scale_by_SL: Optional[bool] = None,  # backward compatibility; if True => "sl"
                 min_pairs: Optional[List[Tuple[int, int]]] = None,
                 percentile: Optional[float] = None,  # e.g., 10.0 for 10th percentile
                 detach_scale: bool = True,
                 clamp_min: float = 1e-6,
                 clamp_max: Optional[float] = None,
                 beta: float = 10.0):
        super().__init__()
        assert len(pairs) > 0, 'Define al menos un par'
        self.pairs = pairs
        self.degree_normalize = degree_normalize
        self.beta = beta
        self.detach_scale = bool(detach_scale)
        self.percentile = percentile
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Back-compat: honor legacy flag
        if scale_by_SL is not None:
            normalization = "sl" if scale_by_SL else ("none" if normalization == "sl" else normalization)
        assert normalization in ("sl", "min_gt", "min_pred", "none"), f'normalization="{normalization}" inválido'
        self.normalization = normalization

        # Degree weights for pairs
        max_k = max(max(i, j) for (i, j) in pairs) + 1
        deg = [0] * max_k
        for (i, j) in pairs:
            deg[i] += 1; deg[j] += 1
        self.register_buffer('deg', torch.tensor(deg, dtype=torch.float32))
        w = []
        for (i, j) in pairs:
            w.append(1.0 / (self.deg[i] + self.deg[j]) if degree_normalize else 1.0)
        w = torch.tensor(w, dtype=torch.float32)
        w = w / (w.sum() + 1e-12)
        self.register_buffer('edge_w', w)

        # SL index (only used if normalization == "sl")
        self.sl_idx = None
        for idx, (i, j) in enumerate(pairs):
            if (i, j) in [(0, 1), (1, 0)]:
                self.sl_idx = idx
                break

        # Build mask of candidate pairs for "min_*" modes
        if min_pairs is None:
            self.register_buffer('min_mask', torch.ones(len(pairs), dtype=torch.bool))
        else:
            mask = torch.zeros(len(pairs), dtype=torch.bool)
            lut = {p: k for k, p in enumerate(pairs)}
            for p in min_pairs:
                if p in lut:
                    mask[lut[p]] = True
                elif (p[1], p[0]) in lut:
                    mask[lut[(p[1], p[0])]] = True
            self.register_buffer('min_mask', mask)

    # ---------- helpers ----------
    def _get_vis_mask(self, data_samples, K: int, device) -> torch.Tensor:
        N = len(data_samples)
        mask = torch.ones((N, K), dtype=torch.bool, device=device)
        for n, ds in enumerate(data_samples):
            gi = getattr(ds, 'gt_instances', None)
            if gi is None:
                continue
            vis_attr = getattr(gi, 'keypoints_visible', None)
            if vis_attr is None:
                continue
            vis = torch.as_tensor(vis_attr, device=device)
            if vis.dtype != torch.bool:
                vis = vis.float() > 0.5
            vis = vis.view(-1)[:K]
            mask[n, :vis.numel()] = vis
        return mask

    def _get_gt_coords_heatmap(self, data_samples, K: int, heatmap_size, device):
        N = len(data_samples)
        W_h, H_h = heatmap_size  # (W, H)
        gt = torch.zeros((N, K, 2), dtype=torch.float32, device=device)
        for n, ds in enumerate(data_samples):
            gi = getattr(ds, 'gt_instances', None); assert gi is not None, 'gt_instances no encontrado'
            kps_attr = getattr(gi, 'keypoints', None); assert kps_attr is not None, 'gt_instances.keypoints no encontrado'
            kps = torch.as_tensor(kps_attr, device=device, dtype=torch.float32)
            if kps.dim() == 3:
                kps = kps[0]
            meta = getattr(ds, 'metainfo', {})
            input_size = meta.get('input_size', None); assert input_size is not None, 'metainfo.input_size requerido'
            W_in, H_in = input_size
            sx, sy = W_h / float(W_in), H_h / float(H_in)
            kps_hm = kps.clone()
            kps_hm[:, 0] *= sx; kps_hm[:, 1] *= sy
            gt[n, :min(K, kps_hm.shape[0]), :] = kps_hm[:K, :]
        return gt

    def _reduce_scale(self, ref_d: Tensor, pair_masks: torch.Tensor) -> Tensor:
        """
        ref_d: (N, M) distances to pick scale from (GT or pred)
        pair_masks: (N, M) bool visibility for each pair
        returns: (N, 1) scale
        """
        # candidate = visible & in min_mask
        cand = pair_masks & self.min_mask.unsqueeze(0)
        if self.percentile is None:
            # hard-min, fill invalid with large number
            big = torch.finfo(ref_d.dtype).max
            masked = torch.where(cand, ref_d, torch.full_like(ref_d, big))
            scale, _ = masked.min(dim=1, keepdim=True)
        else:
            # percentile among valid pairs (fallback to min if none)
            # to avoid heavy ops, approximate via topk on small M (M is #pairs)
            big = torch.finfo(ref_d.dtype).max
            masked = torch.where(cand, ref_d, torch.full_like(ref_d, big))
            # sort ascending and take index at p%
            vals, _ = masked.sort(dim=1)
            # count valid per row
            valid_counts = cand.sum(dim=1, keepdim=True)  # (N,1)
            # index = ceil(p/100 * (count-1))
            idx = (self.percentile / 100.0 * (valid_counts.clamp(min=1) - 1).float()).ceil().long()
            idx = idx.clamp(min=0)
            # gather with broadcasted indices
            scale = torch.gather(vals, 1, idx)
            # when no valid pairs: scale = 1.0
            none_valid = (valid_counts == 0)
            if none_valid.any():
                scale = torch.where(none_valid, torch.ones_like(scale), scale)

        # clamp & eps
        if self.clamp_min is not None:
            scale = scale.clamp_min(self.clamp_min)
        if self.clamp_max is not None:
            scale = scale.clamp_max(self.clamp_max)

        return scale

    # ---------- forward ----------
    def forward(self, pred_heatmaps: torch.Tensor, data_samples: list, heatmap_size=None) -> torch.Tensor:
        device = pred_heatmaps.device
        N, K, H, W = pred_heatmaps.shape

        pred_xy = softargmax2d(pred_heatmaps, beta=self.beta)
        if heatmap_size is None:
            meta0 = getattr(data_samples[0], 'metainfo', {})
            heatmap_size = meta0.get('heatmap_size', (W, H))  # (W, H)
        gt_xy = self._get_gt_coords_heatmap(data_samples, K, heatmap_size, device)

        vis_k = self._get_vis_mask(data_samples, K, device)         # (N, K)
        pair_masks = torch.stack([(vis_k[:, i] & vis_k[:, j]) for (i, j) in self.pairs], dim=1)  # (N, M) bool

        pred_d = torch.stack([pairwise_dist(pred_xy, i, j) for (i, j) in self.pairs], dim=1)     # (N, M)
        gt_d   = torch.stack([pairwise_dist(gt_xy,   i, j) for (i, j) in self.pairs], dim=1)     # (N, M)

        # ---- choose scale ----
        if self.normalization == "sl":
            assert self.sl_idx is not None, 'normalization="sl" requiere que (0,1) esté en pairs'
            scale = gt_d[:, self.sl_idx:self.sl_idx+1]
        elif self.normalization == "min_gt":
            scale = self._reduce_scale(gt_d, pair_masks)
        elif self.normalization == "min_pred":
            scale = self._reduce_scale(pred_d, pair_masks)
        else:  # "none"
            scale = torch.ones((N, 1), device=device, dtype=pred_d.dtype)

        if self.detach_scale:
            scale = scale.detach()

        # avoid div by 0 after detach/clamp
        scale = scale + 1e-6

        pred_d = pred_d / scale
        gt_d   = gt_d   / scale

        Wm = self.edge_w.to(device).unsqueeze(0).expand_as(pred_d)
        mask_f = pair_masks.float()
        residual = (pred_d - gt_d) * mask_f
        loss_mat = (residual ** 2) * Wm

        valid_w = (Wm * mask_f).sum(dim=1) + 1e-12
        loss_vec = torch.zeros(N, device=device, dtype=pred_d.dtype)
        valid_mask = valid_w > 0
        if valid_mask.any():
            loss_vec[valid_mask] = (loss_mat.sum(dim=1)[valid_mask] / valid_w[valid_mask])
        return loss_vec.mean()
