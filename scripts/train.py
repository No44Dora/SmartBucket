from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import TrainingConfig, train_step  # noqa: E402
from src.models import UNetDualHead  # noqa: E402


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
            ext = image_path.suffix.lstrip(".").lower()
            if not ext:
                continue
            target_stem = f"{image_path.stem}__{ext}"
            interior_path = self.interior_dir / f"{target_stem}_interior.png"
            seed_path = self.seed_dir / f"{target_stem}_seed_heatmap.png"
            if interior_path.exists() and seed_path.exists():
                self.samples.append((image_path, interior_path, seed_path))

        if not self.samples:
            raise RuntimeError(
                "No paired samples found. Ensure images/interior_masks/seed_heatmaps share identical filenames."
                " Expected labels like: <name>__<ext>_interior.png and <name>__<ext>_seed_heatmap.png"
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


def save_checkpoint(
    *,
    path: Path,
    model: UNetDualHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    avg_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "avg_loss": avg_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device)

    model = UNetDualHead(
        in_channels=cfg["model"].get("in_channels", 1),
        out_channels=1,
    ).to(device)

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
    output_dir = Path(cfg["train"].get("output_dir", "outputs/checkpoints"))
    best_loss = float("inf")

    for epoch in range(epochs):
        running_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(loader):
            metrics = train_step(model, batch, optimizer, train_cfg, device)
            running_loss += metrics["loss_total"]
            num_steps += 1

            if step % 5 == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"total={metrics['loss_total']:.4f} "
                    f"interior={metrics['loss_interior']:.4f} "
                    f"seed={metrics['loss_seed']:.4f}"
                )

        avg_loss = running_loss / max(num_steps, 1)
        latest_ckpt = output_dir / "latest.pth"
        save_checkpoint(
            path=latest_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            avg_loss=avg_loss,
            config=cfg,
        )
        print(f"Saved checkpoint: {latest_ckpt} (epoch={epoch}, avg_loss={avg_loss:.4f})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = output_dir / "best.pth"
            save_checkpoint(
                path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                avg_loss=avg_loss,
                config=cfg,
            )
            print(f"Updated best checkpoint: {best_ckpt} (best_loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
