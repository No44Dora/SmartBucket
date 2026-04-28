from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss（按 batch 计算后取均值）。"""
    pred = pred.contiguous()
    target = target.contiguous()

    # 计算每个样本的交集/并集
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def interior_loss(pred: torch.Tensor, target: torch.Tensor, bce_weight: float = 1.0, dice_weight: float = 1.0) -> torch.Tensor:
    """Interior 分支损失：BCE + Dice。"""
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dice
