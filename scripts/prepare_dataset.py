"""彩色平涂图像 -> 区域实例标签 预处理脚本。"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PreprocessConfig:
    """彩色平涂图像预处理配置。"""

    num_clusters: int = 24
    sample_size: int = 20000
    kmeans_attempts: int = 1
    merge_lab_threshold: float = 12.0
    min_region_area: int = 128
    bilateral_d: int = 5
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 5.0
    enable_smoothing: bool = True
    line_gray_threshold: int = 55
    ignore_line_pixels: bool = True
    morph_kernel_size: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将彩色平涂图像转换为区域实例标签（region ID map）。"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入图像路径或目录")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--num-clusters", type=int, default=24, help="颜色聚类数")
    parser.add_argument("--sample-size", type=int, default=20000, help="KMeans 训练采样像素数")
    parser.add_argument("--kmeans-attempts", type=int, default=1, help="KMeans 重试次数")
    parser.add_argument(
        "--merge-lab-threshold",
        type=float,
        default=12.0,
        help="Lab 空间颜色中心合并阈值",
    )
    parser.add_argument("--min-region-area", type=int, default=128, help="最小区域像素数")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fast", "quality", "custom"],
        default="fast",
        help="fast/quality 使用预设参数，custom 完全使用命令行参数",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="并行进程数，0 表示串行")
    parser.add_argument("--save-vis", action="store_true", help="是否保存可视化结果")
    parser.add_argument(
        "--vis-max-images",
        type=int,
        default=20,
        help="最多保存可视化的图像数量，仅在 --save-vis 时生效",
    )
    parser.add_argument(
        "--max-vis-instances",
        type=int,
        default=6,
        help="每张图可视化的最多实例数量",
    )
    return parser.parse_args()


def collect_images(input_path: Path) -> list[Path]:
    """收集输入图像列表（支持单图或目录）。"""
    if input_path.is_file():
        return [input_path]

    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in input_path.iterdir() if p.suffix.lower() in suffixes])


def read_color_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取图像: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def smooth_before_clustering(image_rgb: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not cfg.enable_smoothing:
        return image_rgb
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    smoothed = cv2.bilateralFilter(
        bgr,
        d=cfg.bilateral_d,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space,
    )
    return cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)


def estimate_line_mask(image_rgb: np.ndarray, gray_threshold: int = 55) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return gray < gray_threshold


def quantize_colors_kmeans_sampled(
    image_rgb: np.ndarray,
    k: int,
    sample_size: int,
    attempts: int,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """在 Lab 空间采样训练 KMeans，再对全图做最近中心分配。"""

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    if valid_mask is not None:
        valid_flat = valid_mask.reshape(-1)
        train_candidates = pixels[valid_flat]
    else:
        valid_flat = None
        train_candidates = pixels

    num_candidates = int(train_candidates.shape[0])
    if num_candidates == 0:
        raise ValueError("有效像素为空，无法进行 KMeans")
    if k < 1:
        raise ValueError(f"--num-clusters 必须 >= 1，当前值: {k}")

    effective_k = min(k, num_candidates)
    if num_candidates > sample_size:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(num_candidates, sample_size, replace=False)
        train_pixels = train_candidates[sample_idx]
    else:
        train_pixels = train_candidates

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.5,
    )
    _compactness, _sample_labels, centers = cv2.kmeans(
        data=train_pixels,
        K=effective_k,
        bestLabels=None,
        criteria=criteria,
        attempts=max(1, attempts),
        flags=cv2.KMEANS_PP_CENTERS,
    )

    diff = pixels[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    labels = np.argmin(dist2, axis=1).astype(np.int32).reshape(image_rgb.shape[:2])

    centers_lab_u8 = np.clip(centers, 0, 255).astype(np.uint8)[None, :, :]
    centers_rgb = cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2RGB)[0]
    quantized = centers_rgb[labels]

    if valid_flat is not None:
        labels_flat = labels.reshape(-1)
        labels_flat[~valid_flat] = -1
        labels = labels_flat.reshape(labels.shape)

    return quantized, labels, centers_rgb, centers, int(train_pixels.shape[0])


def merge_similar_colors_lab(
    labels: np.ndarray,
    centers_lab: np.ndarray,
    distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按 Lab 空间中心距离合并颜色。"""

    centers = centers_lab.astype(np.float32)
    k = int(len(centers))
    if k == 0:
        return labels.copy(), np.zeros((0, 3), dtype=np.uint8)

    parent = np.arange(k, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(k):
        for j in range(i + 1, k):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            if d <= distance_threshold:
                union(i, j)

    roots = np.array([find(i) for i in range(k)], dtype=np.int32)
    unique_roots = np.unique(roots)
    root_to_gid = {int(r): idx for idx, r in enumerate(unique_roots)}

    label_to_group = np.array([root_to_gid[int(r)] for r in roots], dtype=np.int32)
    merged_labels = np.full_like(labels, fill_value=-1)
    valid_mask = labels >= 0
    merged_labels[valid_mask] = label_to_group[labels[valid_mask]]

    merged_centers_lab = np.array(
        [centers[roots == r].mean(axis=0) for r in unique_roots],
        dtype=np.float32,
    )
    centers_lab_u8 = np.clip(merged_centers_lab, 0, 255).astype(np.uint8)[None, :, :]
    merged_palette_rgb = cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2RGB)[0]

    return merged_labels, merged_palette_rgb


def build_region_instances(
    merged_labels: np.ndarray,
    min_region_area: int,
    morph_kernel_size: int,
) -> np.ndarray:
    """对每种颜色做连通域分析，并分配全局 region ID。"""

    h, w = merged_labels.shape
    region_map = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    for color_id in np.unique(merged_labels):
        if color_id < 0:
            continue
        mask = (merged_labels == color_id).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num, comp_ids, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for comp_idx in range(1, num):
            area = int(stats[comp_idx, cv2.CC_STAT_AREA])
            if area < min_region_area:
                continue
            region_map[comp_ids == comp_idx] = next_id
            next_id += 1

    return region_map


def labels_to_color(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid_mask = labels >= 0
    canvas[valid_mask] = palette[labels[valid_mask]]
    return canvas


def region_to_random_color(region_map: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    max_region = int(region_map.max())
    palette = np.zeros((max_region + 1, 3), dtype=np.uint8)
    if max_region > 0:
        palette[1:] = rng.integers(32, 256, size=(max_region, 3), dtype=np.uint8)
    return palette[region_map]


def save_visualizations(
    image_rgb: np.ndarray,
    quantized_rgb: np.ndarray,
    merged_rgb: np.ndarray,
    region_map: np.ndarray,
    out_prefix: Path,
    max_instances: int,
) -> None:
    region_color = region_to_random_color(region_map)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input")
    axes[1].imshow(quantized_rgb)
    axes[1].set_title("KMeans Quantized")
    axes[2].imshow(merged_rgb)
    axes[2].set_title("Merged by Lab distance")
    axes[3].imshow(region_color)
    axes[3].set_title("Region Instance IDs")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_pipeline.png"), dpi=160)
    plt.close(fig)

    region_ids = [rid for rid in np.unique(region_map) if rid > 0][:max_instances]
    if not region_ids:
        return

    fig, axes = plt.subplots(len(region_ids), 2, figsize=(7, 3 * len(region_ids)))
    if len(region_ids) == 1:
        axes = np.array([axes])

    for row, rid in enumerate(region_ids):
        mask = region_map == rid
        before = image_rgb.copy()
        before[~mask] = (before[~mask] * 0.15).astype(np.uint8)

        after = np.zeros_like(image_rgb)
        after[mask] = [255, 255, 255]

        axes[row, 0].imshow(before)
        axes[row, 0].set_title(f"Before: region {rid}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(after)
        axes[row, 1].set_title(f"After: instance mask {rid}")
        axes[row, 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_instances.png"), dpi=160)
    plt.close(fig)


def process_one_image(
    path: Path,
    output_dir: Path,
    cfg: PreprocessConfig,
    max_instances: int,
    save_vis: bool,
) -> None:
    start_time = time.perf_counter()
    image_rgb = read_color_image(path)
    image_for_cluster = smooth_before_clustering(image_rgb, cfg)

    line_mask = estimate_line_mask(image_rgb, cfg.line_gray_threshold) if cfg.ignore_line_pixels else None
    quantized_rgb, cluster_labels, _centers_rgb, centers_lab, sampled_count = quantize_colors_kmeans_sampled(
        image_for_cluster,
        cfg.num_clusters,
        cfg.sample_size,
        cfg.kmeans_attempts,
        valid_mask=None if line_mask is None else ~line_mask,
    )

    merged_labels, merged_palette = merge_similar_colors_lab(
        cluster_labels,
        centers_lab,
        cfg.merge_lab_threshold,
    )
    if line_mask is not None:
        merged_labels[line_mask] = -1

    merged_rgb = labels_to_color(merged_labels, merged_palette)
    region_map = build_region_instances(merged_labels, cfg.min_region_area, cfg.morph_kernel_size)

    if int(region_map.max()) > int(np.iinfo(np.uint16).max):
        raise ValueError(f"region id 超过 uint16 上限: {int(region_map.max())}")

    safe_suffix = path.suffix.lower().lstrip(".") or "nosuffix"
    stem_out = output_dir / f"{path.stem}__{safe_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(stem_out.with_name(stem_out.name + "_region_id.png")),
        region_map.astype(np.uint16),
    )

    if save_vis:
        save_visualizations(
            image_rgb=image_rgb,
            quantized_rgb=quantized_rgb,
            merged_rgb=merged_rgb,
            region_map=region_map,
            out_prefix=stem_out,
            max_instances=max_instances,
        )

    metadata = {
        "source": str(path),
        "shape": list(image_rgb.shape),
        "num_clusters_initial": int(cfg.num_clusters),
        "kmeans_sample_size": int(cfg.sample_size),
        "kmeans_sampled_pixels": int(sampled_count),
        "kmeans_attempts": int(cfg.kmeans_attempts),
        "num_colors_after_merge": int(len(np.unique(merged_labels[merged_labels >= 0]))),
        "num_regions": int(region_map.max()),
        "merge_lab_threshold": float(cfg.merge_lab_threshold),
        "min_region_area": int(cfg.min_region_area),
        "line_pixels_ignored": bool(cfg.ignore_line_pixels),
        "processing_seconds": float(time.perf_counter() - start_time),
        "save_vis": bool(save_vis),
    }
    with stem_out.with_name(stem_out.name + "_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def build_config(args: argparse.Namespace) -> PreprocessConfig:
    cfg = PreprocessConfig(
        num_clusters=args.num_clusters,
        sample_size=args.sample_size,
        kmeans_attempts=args.kmeans_attempts,
        merge_lab_threshold=args.merge_lab_threshold,
        min_region_area=args.min_region_area,
    )

    if args.mode == "fast":
        cfg.sample_size = 15000
        cfg.kmeans_attempts = 1
        cfg.min_region_area = max(cfg.min_region_area, 128)
        cfg.enable_smoothing = False
    elif args.mode == "quality":
        cfg.sample_size = max(cfg.sample_size, 30000)
        cfg.kmeans_attempts = max(cfg.kmeans_attempts, 2)
        cfg.enable_smoothing = True
        cfg.merge_lab_threshold = args.merge_lab_threshold

    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    images = collect_images(args.input)
    if not images:
        raise ValueError(f"未在输入路径找到可处理图像: {args.input}")

    vis_set: set[Path] = set()
    if args.save_vis and args.vis_max_images > 0:
        vis_set = set(images[: min(args.vis_max_images, len(images))])

    if args.num_workers > 0:
        max_workers = min(args.num_workers, os.cpu_count() or args.num_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    process_one_image,
                    path=img_path,
                    output_dir=args.output,
                    cfg=cfg,
                    max_instances=args.max_vis_instances,
                    save_vis=img_path in vis_set,
                )
                for img_path in images
            ]
            for f in futures:
                f.result()
    else:
        for img_path in images:
            process_one_image(
                path=img_path,
                output_dir=args.output,
                cfg=cfg,
                max_instances=args.max_vis_instances,
                save_vis=img_path in vis_set,
            )

    product_text = "*_region_id.png, *_meta.json"
    if args.save_vis and vis_set:
        product_text += ", (subset) *_pipeline.png, *_instances.png"

    print(
        "完成处理:\n"
        f"- 图像数量: {len(images)}\n"
        f"- 可视化数量: {len(vis_set)}\n"
        f"- 输出目录: {args.output}\n"
        f"- 产物: {product_text}"
    )


if __name__ == "__main__":
    main()
