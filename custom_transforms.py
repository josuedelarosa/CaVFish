# custom_transforms.py
from __future__ import annotations
import math
import random
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mmengine.registry import TRANSFORMS
from mmcv.transforms.base import BaseTransform

# -----------------------------
# Utility helpers (internal)
# -----------------------------

def _kps_inside_canvas(kps: np.ndarray, vis: np.ndarray, w: int, h: int, margin: int = 0) -> bool:
    """True iff all visible keypoints are inside [margin, w-1-margin] x [margin, h-1-margin]."""
    if kps.size == 0:
        return True
    xmin, ymin = margin, margin
    xmax, ymax = w - 1 - margin, h - 1 - margin
    xy = kps.reshape(-1, 2)
    v = (vis.reshape(-1) > 0)
    if not np.any(v):
        return True
    x_ok = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax)
    y_ok = (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
    return bool(np.all(x_ok[v] & y_ok[v]))


def _warp_affine_points(kps: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine M to keypoints array of shape (..., 2)."""
    orig_shape = kps.shape
    pts = kps.reshape(-1, 2).astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([pts, ones], axis=1)  # (N,3)
    out = (M @ hom.T).T[:, :2]
    return out.reshape(orig_shape)

# -----------------------------
# 1) SafeRotateBackoff
# -----------------------------

@TRANSFORMS.register_module()
class SafeRotateBackoff(BaseTransform):
    """Rotate image & keypoints around image center; halve angle until all visible KPs stay inside.

    Expected MMPose fields before this transform:
      - results['img']: np.ndarray HxWxC
      - results['keypoints']: (N,K,2) float32
      - results['keypoints_visible']: (N,K) float32 or int

    Args:
        max_degree (float): initial angle sampled from U(-max_degree, +max_degree). Default 20.
        p (float): probability to attempt rotation. Default 1.0.
        max_iters (int): max number of backoff halvings. Default 10.
        border_value (tuple[int,int,int]): BGR fill value. Default (0,0,0).
        clip_margin (int): extra margin to keep KPs from borders. Default 0.
        deterministic (bool): if True, uses +max_degree for debugging. Default False.
    """

    def __init__(
        self,
        max_degree: float = 20.0,
        p: float = 1.0,
        max_iters: int = 10,
        border_value: Tuple[int, int, int] = (0, 0, 0),
        clip_margin: int = 0,
        deterministic: bool = False,
    ):
        self.max_degree = float(max_degree)
        self.p = float(p)
        self.max_iters = int(max_iters)
        self.border_value = tuple(int(x) for x in border_value)
        self.clip_margin = int(clip_margin)
        self.deterministic = bool(deterministic)

    def transform(self, results: dict) -> dict:
        if random.random() > self.p:
            return results

        img: np.ndarray = results['img']
        h, w = img.shape[:2]

        kps: np.ndarray = results['keypoints']          # (N,K,2)
        vis: np.ndarray = results['keypoints_visible']  # (N,K)

        # sample initial angle
        a = self.max_degree if self.deterministic else random.uniform(-self.max_degree, +self.max_degree)
        cx, cy = (w / 2.0, h / 2.0)
        used_angle = 0.0

        for _ in range(self.max_iters + 1):
            M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
            kps_rot = _warp_affine_points(kps, M)
            if _kps_inside_canvas(kps_rot, vis, w, h, margin=self.clip_margin):
                img_rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=self.border_value)
                results['img'] = img_rot
                results['keypoints'] = kps_rot
                results['rotation'] = float(a)
                used_angle = float(a)
                break
            a *= 0.5  # back off toward 0
        else:
            # no safe angle found; keep original
            results['rotation'] = 0.0

        # small stat for optional logging
        results.setdefault('augment_stats', {})
        results['augment_stats']['safe_rotate_used_deg'] = used_angle
        return results

# -----------------------------
# 2) LocalGridMaskKPs
# -----------------------------

@TRANSFORMS.register_module()
class LocalGridMaskKPs(BaseTransform):
    """Apply a local GridMask inside a square window around selected visible keypoints.

    This is image-only (does not move keypoints).

    Args:
        target_indices (Sequence[int]): eligible keypoint indices.
        box_size (int): square ROI side length centered at each KP. Default 96.
        grid_d (int): grid period in pixels. Default 8.
        ratio (float): black-band fraction per grid cell in [0,1]. Default 0.25.
        angle_deg (float): optional grid rotation. Default 0.0.
        per_kp_prob (float): independent P(apply) per eligible KP. Default 0.25.
        choose_k_range (Tuple[int,int] | None): if set, pick K∈[L,R] KPs per image; overrides per_kp_prob.
        p_img (float): probability to apply the transform to the image. Default 0.5.
    """

    def __init__(
        self,
        target_indices: Sequence[int],
        box_size: int = 96,
        grid_d: int = 8,
        ratio: float = 0.25,
        angle_deg: float = 0.0,
        per_kp_prob: float = 0.25,
        choose_k_range: Optional[Tuple[int, int]] = None,
        p_img: float = 0.5,
    ):
        self.targets = tuple(int(i) for i in target_indices)
        self.box_size = int(box_size)
        self.grid_d = int(grid_d)
        self.ratio = float(ratio)
        self.angle_deg = float(angle_deg)
        self.per_kp_prob = float(per_kp_prob)
        self.choose_k_range = tuple(choose_k_range) if choose_k_range is not None else None
        self.p_img = float(p_img)

    def _apply_gridmask_to_roi(self, roi: np.ndarray) -> np.ndarray:
        rh, rw = roi.shape[:2]
        if rh <= 1 or rw <= 1:
            return roi
        mask = np.ones((rh, rw), dtype=np.uint8)
        band = max(1, int(self.ratio * self.grid_d))

        # horizontal & vertical bands
        for y in range(0, rh, self.grid_d):
            mask[y:min(rh, y + band), :] = 0
        for x in range(0, rw, self.grid_d):
            mask[:, x:min(rw, x + band)] = 0

        if abs(self.angle_deg) > 1e-6:
            M = cv2.getRotationMatrix2D((rw / 2.0, rh / 2.0), self.angle_deg, 1.0)
            mask = cv2.warpAffine(mask, M, (rw, rh), flags=cv2.INTER_NEAREST, borderValue=1)

        out = roi.copy()
        out[mask == 0] = 0
        return out

    def transform(self, results: dict) -> dict:
        if random.random() > self.p_img:
            return results

        img: np.ndarray = results['img']
        h, w = img.shape[:2]

        kps: np.ndarray = results['keypoints']          # (N,K,2)
        vis: np.ndarray = results['keypoints_visible']  # (N,K)
        assert kps.ndim == 3 and vis.ndim == 2, "Expect (N,K,2) keypoints and (N,K) visibility"

        # assume top-down single instance; if multiple, operate on the first
        kp_xy = kps[0] if kps.shape[0] >= 1 else np.zeros((0, 2), dtype=np.float32)
        kp_vis = vis[0] if vis.shape[0] >= 1 else np.zeros((0,), dtype=np.float32)

        eligible = [i for i in self.targets if i < kp_xy.shape[0] and kp_vis[i] > 0]
        chosen: List[int] = []

        if self.choose_k_range is not None and len(eligible) > 0:
            L, R = self.choose_k_range
            k = max(0, min(len(eligible), random.randint(L, R)))
            chosen = random.sample(eligible, k)
        else:
            for i in eligible:
                if random.random() < self.per_kp_prob:
                    chosen.append(i)

        out = img.copy()
        half = self.box_size // 2
        masked_count = 0

        for idx in chosen:
            cx, cy = int(round(kp_xy[idx, 0])), int(round(kp_xy[idx, 1]))
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = out[y1:y2, x1:x2]
            out[y1:y2, x1:x2] = self._apply_gridmask_to_roi(roi)
            masked_count += 1

        results['img'] = out
        results.setdefault('augment_stats', {})
        results['augment_stats']['gridmask_regions'] = masked_count
        return results

# -----------------------------
# 3) AlbumentationsColorNoise (image-only)
# -----------------------------

try:
    import albumentations as A
except Exception:
    A = None

@TRANSFORMS.register_module()
class AlbumentationsColorNoise(BaseTransform):
    """Simple Albumentations wrapper for GaussNoise + ColorJitter (image-only).

    Args:
        noise_var (tuple[float,float]): GaussNoise var_limit. Default (5.0, 20.0).
        p_noise (float): probability for noise. Default 0.25.
        brightness (float): ColorJitter brightness. Default 0.25.
        contrast (float): ColorJitter contrast. Default 0.25.
        saturation (float): ColorJitter saturation. Default 0.20.
        hue (float): ColorJitter hue (0..0.5). Default 0.03.
        p_color (float): probability for color jitter. Default 0.70.
    """

    def __init__(
        self,
        noise_var: Tuple[float, float] = (5.0, 20.0),
        p_noise: float = 0.25,
        brightness: float = 0.25,
        contrast: float = 0.25,
        saturation: float = 0.20,
        hue: float = 0.03,
        p_color: float = 0.70,
    ):
        if A is None:
            raise ImportError(
                "Albumentations is not installed. "
                "Install it with: pip install albumentations opencv-python-headless"
            )
        self.albu = A.Compose([
            A.GaussNoise(var_limit=noise_var, p=p_noise),
            A.ColorJitter(brightness=brightness, contrast=contrast,
                          saturation=saturation, hue=hue, p=p_color),
        ])

    def transform(self, results: dict) -> dict:
        img = results['img']
        out = self.albu(image=img)
        results['img'] = out['image']
        return results
