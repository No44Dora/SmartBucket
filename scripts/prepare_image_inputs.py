from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class PreprocessConfig:
    target_size: int
    square_tolerance: float = 0.15
    foreground_threshold: int = 245
    bbox_margin_ratio: float = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop and resize raw color images to fixed-size training inputs.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Raw color image folder")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder of resized/cropped color images")
    parser.add_argument("--target-size", type=int, default=256, help="Output square size")
    parser.add_argument(
        "--square-tolerance",
        type=float,
        default=0.15,
        help="Treat image as near-square when abs(w/h - 1) <= tolerance",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=int,
        default=245,
        help="Foreground threshold used when detecting main subject",
    )
    parser.add_argument(
        "--bbox-margin-ratio",
        type=float,
        default=0.08,
        help="Margin ratio expanded from detected foreground box",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".webp"],
        help="Image suffixes to process",
    )
    return parser.parse_args()


def find_foreground_bbox(gray: np.ndarray, threshold: int) -> tuple[int, int, int, int] | None:
    mask = gray < threshold
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    margin = int(max(bw, bh) * margin_ratio)
    x1 = clamp(x1 - margin, 0, width - 1)
    y1 = clamp(y1 - margin, 0, height - 1)
    x2 = clamp(x2 + margin, 0, width - 1)
    y2 = clamp(y2 + margin, 0, height - 1)
    return x1, y1, x2, y2


def fit_crop_to_square(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    side = max(x2 - x1 + 1, y2 - y1 + 1)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    nx1 = clamp(cx - side // 2, 0, max(0, width - side))
    ny1 = clamp(cy - side // 2, 0, max(0, height - side))
    nx2 = clamp(nx1 + side - 1, 0, width - 1)
    ny2 = clamp(ny1 + side - 1, 0, height - 1)
    return nx1, ny1, nx2, ny2


def preprocess_color_image(image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
    width, height = image.size
    ratio = width / height

    if abs(ratio - 1.0) <= cfg.square_tolerance:
        return image.resize((cfg.target_size, cfg.target_size), Image.Resampling.BICUBIC)

    gray = np.array(image.convert("L"))
    bbox = find_foreground_bbox(gray, cfg.foreground_threshold)

    if bbox is None:
        side = min(width, height)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        crop_box = (x1, y1, x1 + side, y1 + side)
    else:
        expanded = expand_bbox(bbox, width, height, cfg.bbox_margin_ratio)
        x1, y1, x2, y2 = fit_crop_to_square(expanded, width, height)
        crop_box = (x1, y1, x2 + 1, y2 + 1)

    return image.crop(crop_box).resize((cfg.target_size, cfg.target_size), Image.Resampling.BICUBIC)


def iter_images(input_dir: Path, suffixes: set[str]) -> list[Path]:
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
    return sorted(files)


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        target_size=args.target_size,
        square_tolerance=args.square_tolerance,
        foreground_threshold=args.foreground_threshold,
        bbox_margin_ratio=args.bbox_margin_ratio,
    )

    suffixes = {s.lower() for s in args.suffixes}
    image_paths = iter_images(args.input_dir, suffixes)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        print(f"[WARN] No images found in: {args.input_dir}")
        return

    for src_path in image_paths:
        rel = src_path.relative_to(args.input_dir)
        dst = args.output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        color = Image.open(src_path).convert("RGB")
        color_processed = preprocess_color_image(color, cfg)
        color_processed.save(dst)

    print(f"[INFO] Processed {len(image_paths)} raw color images.")


if __name__ == "__main__":
    main()
