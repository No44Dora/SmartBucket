"""从 region_id 标签图生成 Interior Map 与 Seed Heatmap 真值。"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 prepare_dataset.py 导出的 *_region_id.png 转为 interior/seed_heatmap 监督标签。"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入目录或单个 *_region_id.png")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument(
        "--suffix",
        type=str,
        default="_region_id.png",
        help="输入文件后缀，默认 _region_id.png",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CPU 并行进程数，0 表示串行",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="是否保存可视化（interior 与 heatmap 伪彩）",
    )
    return parser.parse_args()


def collect_region_maps(input_path: Path, suffix: str) -> list[Path]:
    if input_path.is_file():
        if not input_path.name.endswith(suffix):
            raise ValueError(f"输入文件需以 {suffix} 结尾: {input_path}")
        return [input_path]

    files = sorted(p for p in input_path.iterdir() if p.is_file() and p.name.endswith(suffix))
    return files


def read_region_map(path: Path) -> np.ndarray:
    region = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if region is None:
        raise ValueError(f"无法读取 region_id: {path}")
    if region.ndim != 2:
        raise ValueError(f"region_id 必须是单通道标签图: {path}")
    if not np.issubdtype(region.dtype, np.integer):
        raise ValueError(f"region_id dtype 必须是整型: {path}, got {region.dtype}")
    return region.astype(np.int32)


def build_interior_map(region_map: np.ndarray) -> np.ndarray:
    return (region_map > 0).astype(np.uint8)


def build_seed_heatmap(region_map: np.ndarray) -> np.ndarray:
    heatmap = np.zeros(region_map.shape, dtype=np.float32)

    region_ids = np.unique(region_map)
    region_ids = region_ids[region_ids > 0]

    for rid in region_ids:
        mask = (region_map == rid).astype(np.uint8)
        if int(mask.sum()) == 0:
            continue

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        max_dist = float(dist.max())
        if max_dist <= EPS:
            continue

        hi = dist / max_dist
        heatmap = np.maximum(heatmap, hi)

    return np.clip(heatmap, 0.0, 1.0)


def save_outputs(
    region_path: Path,
    output_dir: Path,
    input_suffix: str,
    interior: np.ndarray,
    heatmap: np.ndarray,
    save_vis: bool,
) -> None:
    stem = region_path.name.removesuffix(input_suffix)

    interior_dir = output_dir / "interior_masks"
    heatmap_dir = output_dir / "seed_heatmaps"
    interior_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    interior_u8 = (interior * 255).astype(np.uint8)
    heatmap_u16 = np.round(heatmap * 65535.0).astype(np.uint16)

    cv2.imwrite(str(interior_dir / f"{stem}_interior.png"), interior_u8)
    cv2.imwrite(str(heatmap_dir / f"{stem}_seed_heatmap.png"), heatmap_u16)

    if save_vis:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        heatmap_u8 = np.round(heatmap * 255.0).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(vis_dir / f"{stem}_interior_vis.png"), interior_u8)
        cv2.imwrite(str(vis_dir / f"{stem}_seed_heatmap_vis.png"), heatmap_color)


def process_one(region_path: Path, output_dir: Path, input_suffix: str, save_vis: bool) -> dict:
    start = time.perf_counter()
    region_map = read_region_map(region_path)
    interior = build_interior_map(region_map)
    heatmap = build_seed_heatmap(region_map)
    save_outputs(region_path, output_dir, input_suffix, interior, heatmap, save_vis)

    return {
        "source": str(region_path),
        "shape": list(region_map.shape),
        "num_regions": int(region_map.max()),
        "interior_pixels": int(interior.sum()),
        "heatmap_min": float(heatmap.min()),
        "heatmap_max": float(heatmap.max()),
        "processing_seconds": float(time.perf_counter() - start),
    }


def main() -> None:
    args = parse_args()
    region_files = collect_region_maps(args.input, args.suffix)
    if not region_files:
        raise ValueError(f"未找到任何 region_id 文件: {args.input} (suffix={args.suffix})")

    args.output.mkdir(parents=True, exist_ok=True)

    logs: list[dict] = []
    if args.num_workers > 0:
        max_workers = min(args.num_workers, os.cpu_count() or args.num_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(process_one, p, args.output, args.suffix, args.save_vis)
                for p in region_files
            ]
            for future in futures:
                logs.append(future.result())
    else:
        for p in region_files:
            logs.append(process_one(p, args.output, args.suffix, args.save_vis))

    summary = {
        "num_files": len(logs),
        "output": str(args.output),
        "num_workers": int(args.num_workers),
        "save_vis": bool(args.save_vis),
        "files": logs,
    }
    with (args.output / "supervision_meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "完成监督标签生成:\n"
        f"- 文件数量: {len(logs)}\n"
        f"- 输出目录: {args.output}\n"
        f"- CPU 并行进程: {args.num_workers}\n"
        "- 产物: interior_masks/*_interior.png, seed_heatmaps/*_seed_heatmap.png, supervision_meta.json"
    )


if __name__ == "__main__":
    main()
