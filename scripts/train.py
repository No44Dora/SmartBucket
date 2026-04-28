from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.engine import TrainingConfig, train_step
from src.models import UNetDualHead


class DummyLineartDataset(Dataset):
    """阶段B占位数据集。

    说明：
    - 当前仓库尚未接入真实数据流水线（阶段A）
    - 该数据集仅用于验证模型/损失/训练循环能否跑通
    """

    def __init__(self, size: int = 16, image_size: int = 256) -> None:
        self.size = size
        self.image_size = image_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        _ = idx
        # 模拟单通道线稿输入
        image = torch.rand(1, self.image_size, self.image_size)
        # 模拟 interior 二值标签
        interior = (torch.rand(1, self.image_size, self.image_size) > 0.5).float()
        # 模拟 seed heatmap（仅在 interior 内有值）
        seed = torch.rand(1, self.image_size, self.image_size) * interior
        return {"image": image, "interior": interior, "seed": seed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual-head U-Net (stage B)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """读取 YAML 配置。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device)

    # 构建模型
    model = UNetDualHead(
        in_channels=cfg["model"].get("in_channels", 1),
        out_channels=1,
    ).to(device)

    # 训练损失权重配置
    train_cfg = TrainingConfig(
        lambda_interior=cfg["loss"].get("lambda_interior", 1.0),
        lambda_seed=cfg["loss"].get("lambda_seed", 1.0),
        alpha=cfg["loss"].get("alpha", 3.0),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["optim"].get("lr", 1e-3))

    # 当前先使用占位数据，后续替换成真实 LineartDataset
    dataset = DummyLineartDataset(
        size=cfg["data"].get("size", 16),
        image_size=cfg["data"].get("image_size", 256),
    )
    loader = DataLoader(dataset, batch_size=cfg["data"].get("batch_size", 2), shuffle=True)

    epochs = cfg["train"].get("epochs", 1)
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            metrics = train_step(model, batch, optimizer, train_cfg, device)
            if step % 5 == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"total={metrics['loss_total']:.4f} "
                    f"interior={metrics['loss_interior']:.4f} "
                    f"seed={metrics['loss_seed']:.4f}"
                )


if __name__ == "__main__":
    main()
