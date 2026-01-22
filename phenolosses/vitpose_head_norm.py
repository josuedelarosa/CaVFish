from mmpose.registry import MODELS
from mmpose.models.heads.heatmap_heads.heatmap_head import HeatmapHead
import torch
import torch.nn.functional as F

@MODELS.register_module()
class HeatmapHeadNorm(HeatmapHead):
    """HeatmapHead + optional coordinate-space L1 loss."""
    def __init__(self,
                 loss_coord=None,
                 alpha_coord: float = 0.0,
                 beta_softarg: float = 10.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_coord = MODELS.build(loss_coord) if loss_coord else None
        self.alpha_coord = float(alpha_coord)
        self.beta_softarg = float(beta_softarg)

    def _softargmax2d(self, heatmaps, beta: float):
        """Convert heatmaps → coordinates with soft-argmax."""
        B, K, H, W = heatmaps.shape
        y = torch.linspace(0, H - 1, H, device=heatmaps.device)
        x = torch.linspace(0, W - 1, W, device=heatmaps.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        h = heatmaps.reshape(B * K, H * W)
        p = F.softmax(beta * h, dim=-1)
        exp_x = (p * xx.reshape(-1)).sum(-1)
        exp_y = (p * yy.reshape(-1)).sum(-1)
        return torch.stack([exp_x, exp_y], dim=-1).reshape(B, K, 2)

    def loss_by_feat(self, preds, data_samples=None):
        # --- main heatmap MSE loss ---
        losses = super().loss_by_feat(preds, data_samples)
        if not (self.loss_coord and self.alpha_coord > 0):
            return losses

        pred = preds[-1] if isinstance(preds, (list, tuple)) else preds  # [B,K,H,W]
        pred_xy = self._softargmax2d(pred, beta=self.beta_softarg)        # [B,K,2]

        # get ground-truth coords and visibility
        gt_xy_list, vis_list = [], []
        for ds in data_samples:
            inst = ds.gt_instances
            gt = inst.keypoints.clone()                # [K,2] in input coords
            vis = inst.keypoints_visible.clone().float().squeeze(-1)
            in_w, in_h = ds.metainfo['input_size']
            hm_w, hm_h = ds.metainfo['heatmap_size']
            gt[..., 0] *= hm_w / in_w
            gt[..., 1] *= hm_h / in_h
            gt_xy_list.append(gt)
            vis_list.append(vis)
        gt_xy = torch.stack(gt_xy_list).to(pred.device)
        vis = torch.stack(vis_list).to(pred.device)

        # --- coordinate-space L1 ---
        l1 = self.loss_coord(pred_xy, gt_xy)           # [B,K,2]
        l1 = (l1 * vis.unsqueeze(-1)).mean()           # mask invisible points
        losses['loss_coord'] = self.alpha_coord * l1
        return losses