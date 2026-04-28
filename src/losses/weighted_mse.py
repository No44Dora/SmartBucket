from __future__ import annotations

import torch


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 3.0) -> torch.Tensor:
    """Seed Heatmap 回归损失：Weighted MSE。

    权重定义：w = 1 + alpha * target
    目标是增大区域中心（高 target）的监督强度。
    """
    weight = 1.0 + alpha * target
    return (weight * (pred - target) ** 2).mean()
