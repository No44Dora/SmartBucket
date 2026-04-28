from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class PreprocessConfig:
    target_size: int
    square_tolerance: float = 0.15
    foreground_threshold: int = 245
    bbox_margin_ratio: float = 0.08
    line_threshold: int = 42
    min_region_area: int = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw color images, then derive lineart and region instances.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Raw color image folder")
    parser.add_argument("--color-output-dir", type=Path, required=True, help="Output folder of resized color images")
    parser.add_argument("--lineart-output-dir", type=Path, required=True, help="Output folder of extracted lineart")
    parser.add_argument(
        "--instance-output-dir",
        type=Path,
        required=True,
        help="Output folder of extracted region instance labels",
    )
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
        "--line-threshold",
        type=int,
        default=42,
        help="Threshold for binarizing edge response into lineart",
    )
    parser.add_argument(
        "--min-region-area",
        type=int,
        default=64,
        help="Minimum connected region area to keep in instance map",
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


def extract_lineart(color_image: Image.Image, line_threshold: int) -> Image.Image:
    gray = color_image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.uint8)

    line_mask = edge_arr > line_threshold
    lineart = np.where(line_mask, 0, 255).astype(np.uint8)
    return Image.fromarray(lineart, mode="L")


def build_instance_map(lineart: Image.Image, min_region_area: int) -> np.ndarray:
    line = np.array(lineart, dtype=np.uint8)
    walkable = line > 127
    h, w = walkable.shape

    visited = np.zeros((h, w), dtype=bool)
    labels = np.zeros((h, w), dtype=np.uint16)
    next_label = 1

    for y in range(h):
        for x in range(w):
            if not walkable[y, x] or visited[y, x]:
                continue

            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            touches_border = False

            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))

                if cy == 0 or cy == h - 1 or cx == 0 or cx == w - 1:
                    touches_border = True

                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and walkable[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if touches_border or len(component) < min_region_area:
                continue

            for py, px in component:
                labels[py, px] = next_label
            next_label += 1

    return labels


def iter_images(input_dir: Path, suffixes: set[str]) -> list[Path]:
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
    return sorted(files)


def save_instance_map(instance_map: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(instance_map, mode="I;16")
    image.save(path.with_suffix(".png"))


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        target_size=args.target_size,
        square_tolerance=args.square_tolerance,
        foreground_threshold=args.foreground_threshold,
        bbox_margin_ratio=args.bbox_margin_ratio,
        line_threshold=args.line_threshold,
        min_region_area=args.min_region_area,
    )

    suffixes = {s.lower() for s in args.suffixes}
    image_paths = iter_images(args.input_dir, suffixes)

    args.color_output_dir.mkdir(parents=True, exist_ok=True)
    args.lineart_output_dir.mkdir(parents=True, exist_ok=True)
    args.instance_output_dir.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        print(f"[WARN] No images found in: {args.input_dir}")
        return

    for src_path in image_paths:
        rel = src_path.relative_to(args.input_dir)
        color_dst = args.color_output_dir / rel
        line_dst = args.lineart_output_dir / rel.with_suffix(".png")
        inst_dst = args.instance_output_dir / rel.with_suffix(".png")

        color_dst.parent.mkdir(parents=True, exist_ok=True)
        line_dst.parent.mkdir(parents=True, exist_ok=True)

        color = Image.open(src_path).convert("RGB")
        color_processed = preprocess_color_image(color, cfg)
        color_processed.save(color_dst)

        lineart = extract_lineart(color_processed, cfg.line_threshold)
        lineart.save(line_dst)

        instance_map = build_instance_map(lineart, cfg.min_region_area)
        save_instance_map(instance_map, inst_dst)

    print(f"[INFO] Processed {len(image_paths)} images from raw color images.")


if __name__ == "__main__":
    main()
