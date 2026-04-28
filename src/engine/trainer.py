from __future__ import annotations

from dataclasses import dataclass

import torch

from src.losses import interior_loss, weighted_mse_loss


@dataclass
class TrainingConfig:
    """训练阶段超参数配置。"""

    lambda_interior: float = 1.0
    lambda_seed: float = 1.0
    alpha: float = 3.0


def compute_total_loss(
    interior_pred: torch.Tensor,
    seed_pred: torch.Tensor,
    interior_gt: torch.Tensor,
    seed_gt: torch.Tensor,
    config: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算总损失并返回便于日志记录的分项指标。"""
    loss_interior = interior_loss(interior_pred, interior_gt)
    loss_seed = weighted_mse_loss(seed_pred, seed_gt, alpha=config.alpha)

    # L = λ1 * L_interior + λ2 * L_seed
    total = config.lambda_interior * loss_interior + config.lambda_seed * loss_seed
    metrics = {
        "loss_total": float(total.detach().cpu()),
        "loss_interior": float(loss_interior.detach().cpu()),
        "loss_seed": float(loss_seed.detach().cpu()),
    }
    return total, metrics


def train_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    """单个 iteration 的标准训练流程。"""
    model.train()

    # 读取并搬运 batch 到目标设备
    images = batch["image"].to(device)
    interior_gt = batch["interior"].to(device)
    seed_gt = batch["seed"].to(device)

    optimizer.zero_grad(set_to_none=True)
    interior_pred, seed_pred = model(images)

    total_loss, metrics = compute_total_loss(interior_pred, seed_pred, interior_gt, seed_gt, config)
    total_loss.backward()
    optimizer.step()
    return metrics
