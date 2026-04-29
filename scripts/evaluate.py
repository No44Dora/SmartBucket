from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import UNetDualHead  # noqa: E402
from src.postprocess import run_postprocess  # noqa: E402


class LineartTestDataset(Dataset):
    """读取 processed/test 目录结构下的线稿测试数据。"""

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
            raise RuntimeError("No valid test samples found.")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_gray(self, path: Path, *, nearest: bool, preserve_u16: bool = False) -> np.ndarray:
        image_pil = Image.open(path)
        if preserve_u16:
            image_np = np.asarray(image_pil)
            if image_np.dtype == np.uint16:
                image = Image.fromarray(image_np, mode="I;16")
            else:
                image = image_pil.convert("L")
        else:
            image = image_pil.convert("L")
        resample = Image.NEAREST if nearest else Image.BILINEAR
        image = image.resize((self.image_size, self.image_size), resample=resample)
        image_np = np.asarray(image)
        if image_np.dtype == np.uint16:
            return image_np.astype(np.float32) / 65535.0
        return image_np.astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_path, interior_path, seed_path = self.samples[idx]
        image = self._read_gray(image_path, nearest=False)
        interior = self._read_gray(interior_path, nearest=True)
        seed = self._read_gray(seed_path, nearest=True, preserve_u16=True)

        return {
            "name": image_path.stem,
            "image": torch.from_numpy(image).unsqueeze(0),
            "interior": torch.from_numpy(interior).unsqueeze(0),
            "seed": torch.from_numpy(seed).unsqueeze(0),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model and export visualization")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-vis", type=int, default=8)
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="额外保存不做伪彩/后处理的原始 heatmap（16-bit PNG + .npy）。",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _random_color_map(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(label_map)
    rng = np.random.default_rng(42)
    for rid in ids:
        if rid == 0:
            continue
        rgb = rng.integers(0, 256, size=3, dtype=np.uint8)
        color[label_map == rid] = rgb
    return color


def make_vis_row(sample: dict[str, torch.Tensor | str], interior_pred: np.ndarray, seed_pred: np.ndarray) -> Image.Image:
    image = sample["image"].squeeze().numpy()
    interior_gt = sample["interior"].squeeze().numpy()
    seed_gt = sample["seed"].squeeze().numpy()

    panels = [
        Image.fromarray(_to_u8(image), mode="L"),
        Image.fromarray(_to_u8(interior_gt), mode="L"),
        Image.fromarray(_to_u8(interior_pred), mode="L"),
        Image.fromarray(_to_u8(seed_gt), mode="L"),
        Image.fromarray(_to_u8(seed_pred), mode="L"),
    ]

    w, h = panels[0].size
    canvas = Image.new("L", (w * len(panels), h))
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * w, 0))
    return canvas



def save_final_flat_result(label_map: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    color_map = _random_color_map(label_map)
    Image.fromarray(color_map, mode="RGB").save(output_dir / "final_flat_fill.png")


def save_raw_heatmaps(vis_dir: Path, sample_name: str, seed_gt: np.ndarray, seed_pred: np.ndarray) -> None:
    raw_dir = vis_dir / f"{sample_name}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    seed_gt_clip = np.clip(seed_gt, 0.0, 1.0)
    seed_pred_clip = np.clip(seed_pred, 0.0, 1.0)

    seed_gt_u16 = np.round(seed_gt_clip * 65535.0).astype(np.uint16)
    seed_pred_u16 = np.round(seed_pred_clip * 65535.0).astype(np.uint16)

    Image.fromarray(seed_gt_u16, mode="I;16").save(raw_dir / "seed_gt_raw_u16.png")
    Image.fromarray(seed_pred_u16, mode="I;16").save(raw_dir / "seed_pred_raw_u16.png")
    np.save(raw_dir / "seed_gt_raw.npy", seed_gt_clip.astype(np.float32))
    np.save(raw_dir / "seed_pred_raw.npy", seed_pred_clip.astype(np.float32))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    model = UNetDualHead(
        in_channels=cfg["model"].get("in_channels", 1),
        out_channels=1,
    ).to(device)

    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_root = Path(cfg.get("test", {}).get("root", "processed/test"))
    image_size = cfg["data"].get("image_size", 512)
    dataset = LineartTestDataset(test_root, image_size=image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    vis_dir = Path(cfg.get("test", {}).get("vis_dir", "outputs/eval_vis"))
    vis_dir.mkdir(parents=True, exist_ok=True)
    pp_cfg = cfg.get("postprocess", {})

    mse_interior = 0.0
    mse_seed = 0.0
    count = 0

    with torch.no_grad():
        for sample in loader:
            image = sample["image"].to(device)
            interior_gt = sample["interior"].to(device)
            seed_gt = sample["seed"].to(device)

            interior_pred, seed_pred = model(image)

            mse_interior += torch.mean((interior_pred - interior_gt) ** 2).item()
            mse_seed += torch.mean((seed_pred - seed_gt) ** 2).item()
            count += 1

            if count <= args.max_vis:
                interior_np = interior_pred.squeeze().cpu().numpy()
                seed_np = seed_pred.squeeze().cpu().numpy()
                row = make_vis_row(
                    {
                        "name": sample["name"][0],
                        "image": sample["image"][0].cpu(),
                        "interior": sample["interior"][0].cpu(),
                        "seed": sample["seed"][0].cpu(),
                    },
                    interior_np,
                    seed_np,
                )
                row.save(vis_dir / f"{sample['name'][0]}_comparison.png")
                if args.save_raw:
                    save_raw_heatmaps(
                        vis_dir=vis_dir,
                        sample_name=sample["name"][0],
                        seed_gt=sample["seed"][0].squeeze().cpu().numpy(),
                        seed_pred=seed_np,
                    )

                interior_th = pp_cfg.get("interior_threshold", 0.5)
                smooth_sigma = pp_cfg.get("smooth_sigma", 1.5)
                smooth_kernel = pp_cfg.get("smooth_kernel_size", 7)
                peak_threshold = pp_cfg.get("peak_threshold", 0.4)
                default_min_distance = 8 if image_size <= 256 else 12
                min_distance = pp_cfg.get("min_distance", default_min_distance)
                default_min_area = max(20, int(image_size * image_size * 0.0005))
                min_area = pp_cfg.get("min_area", default_min_area)

                pp_outputs = run_postprocess(
                    interior_pred=interior_pred,
                    seed_pred=seed_pred,
                    interior_threshold=interior_th,
                    smooth_kernel_size=smooth_kernel,
                    smooth_sigma=smooth_sigma,
                    peak_threshold=peak_threshold,
                    min_distance=min_distance,
                    min_area=min_area,
                )
                label_map = pp_outputs["label_map"][0].cpu().numpy().astype(np.int32)
                save_final_flat_result(label_map, vis_dir / f"{sample['name'][0]}_postprocess")

    print(f"Samples: {count}")
    print(f"Interior MSE: {mse_interior / max(count, 1):.6f}")
    print(f"Seed MSE: {mse_seed / max(count, 1):.6f}")
    print(f"Visualization dir: {vis_dir}")


if __name__ == "__main__":
    main()
