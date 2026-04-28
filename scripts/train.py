from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.engine import TrainingConfig, train_step
from src.models import UNetDualHead


class LineartDataset(Dataset):
    """读取 processed/train 目录结构下的线稿训练数据。"""

    def __init__(self, root: Path, image_size: int = 512) -> None:
        self.root = root
        self.image_size = image_size
        self.image_dir = self.root / "images"
        self.interior_dir = self.root / "interior_masks"
        self.seed_dir = self.root / "seed_heatmaps"

        for required in [self.image_dir, self.interior_dir, self.seed_dir]:
            if not required.exists():
                raise FileNotFoundError(f"Missing required directory: {required}")

        self.samples: list[tuple[Path, Path, Path]] = []
        for image_path in sorted(p for p in self.image_dir.iterdir() if p.is_file()):
            interior_path = self.interior_dir / image_path.name
            seed_path = self.seed_dir / image_path.name
            if interior_path.exists() and seed_path.exists():
                self.samples.append((image_path, interior_path, seed_path))

        if not self.samples:
            raise RuntimeError(
                "No paired samples found. Ensure images/interior_masks/seed_heatmaps share identical filenames."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _read_gray(self, path: Path, *, nearest: bool) -> np.ndarray:
        image = Image.open(path).convert("L")
        resample = Image.NEAREST if nearest else Image.BILINEAR
        image = image.resize((self.image_size, self.image_size), resample=resample)
        return np.asarray(image, dtype=np.float32) / 255.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path, interior_path, seed_path = self.samples[idx]
        image = self._read_gray(image_path, nearest=False)
        interior = self._read_gray(interior_path, nearest=True)
        seed = self._read_gray(seed_path, nearest=True)
        interior = (interior > 0.5).astype(np.float32)

        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "interior": torch.from_numpy(interior).unsqueeze(0),
            "seed": torch.from_numpy(seed).unsqueeze(0),
        }


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

    dataset = LineartDataset(
        root=Path(cfg["data"]["root"]),
        image_size=cfg["data"].get("image_size", 512),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"].get("batch_size", 2),
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=cfg["data"].get("pin_memory", False),
    )

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
