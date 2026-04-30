from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import UNetDualHead  # noqa: E402
from src.postprocess import run_postprocess  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单张黑白线稿推理并输出可上色区域")
    parser.add_argument("--config", type=Path, required=True, help="训练配置文件")
    parser.add_argument("--ckpt", type=Path, required=True, help="模型权重文件")
    parser.add_argument("--input", type=Path, required=True, help="黑白线稿路径")
    parser.add_argument("--output", type=Path, default=Path("outputs/infer"), help="输出目录")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备，例如 cpu/cuda")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_lineart(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("L")
    orig_size = image.size
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)
    return tensor, orig_size


def random_color_map(label_map: np.ndarray, seed: int = 42) -> np.ndarray:
    h, w = label_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(label_map)
    rng = np.random.default_rng(seed)

    for rid in ids:
        if rid == 0:
            continue
        color[label_map == rid] = rng.integers(0, 256, size=3, dtype=np.uint8)

    return color


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    image_size = cfg["data"].get("image_size", 512)
    model = UNetDualHead(in_channels=cfg["model"].get("in_channels", 1), out_channels=1).to(device)

    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    image_tensor, orig_size = load_lineart(args.input, image_size=image_size)
    image_tensor = image_tensor.to(device)

    pp_cfg = cfg.get("postprocess", {})

    with torch.no_grad():
        interior_pred, seed_pred = model(image_tensor)
        outputs = run_postprocess(
            interior_pred=interior_pred,
            seed_pred=seed_pred,
            interior_threshold=pp_cfg.get("interior_threshold", 0.5),
            smooth_kernel_size=pp_cfg.get("smooth_kernel_size", 7),
            smooth_sigma=pp_cfg.get("smooth_sigma", 1.5),
            peak_threshold=pp_cfg.get("peak_threshold", 0.4),
            min_distance=pp_cfg.get("min_distance", 12),
            min_area=pp_cfg.get("min_area", 20),
        )

    label_map = outputs["label_map"].squeeze().cpu().numpy().astype(np.uint16)
    color_map = random_color_map(label_map)

    label_img = Image.fromarray(label_map, mode="I;16").resize(orig_size, resample=Image.NEAREST)
    color_img = Image.fromarray(color_map, mode="RGB").resize(orig_size, resample=Image.NEAREST)

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    label_path = args.output / f"{stem}_region_labels.png"
    color_path = args.output / f"{stem}_region_preview.png"

    label_img.save(label_path)
    color_img.save(color_path)

    print(f"Saved region labels: {label_path}")
    print(f"Saved region preview: {color_path}")


if __name__ == "__main__":
    main()
