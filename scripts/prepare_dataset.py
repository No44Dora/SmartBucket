"""彩色平涂图像 -> 区域实例标签 预处理脚本。

使用场景：
1) 将上色参考图或平涂图转换为区域实例 ID 图；
2) 为后续 Interior / Seed Heatmap 监督标签生成提供基础分区；
3) 快速可视化“量化 -> 合并 -> 实例化”效果，便于调参与排查。

核心思路：
- 颜色量化阶段先聚类，减少颜色噪声；
- 颜色合并阶段“优先看色相”，并允许一定明度/饱和度差异，增强对阴影和渐变的鲁棒性；
- 对合并后的每种颜色做连通域分解，最终每个连通块映射为唯一 region ID。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PreprocessConfig:
    """彩色平涂图像预处理配置。"""

    num_clusters: int = 24
    hue_threshold: float = 12.0
    sat_threshold: float = 60.0
    val_threshold: float = 70.0
    min_region_area: int = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将彩色平涂图像转换为区域实例标签（region ID map）。"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入图像路径或目录")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--num-clusters", type=int, default=24, help="颜色聚类数")
    parser.add_argument("--hue-threshold", type=float, default=12.0, help="色相合并阈值")
    parser.add_argument(
        "--sat-threshold",
        type=float,
        default=60.0,
        help="饱和度容忍阈值（允许渐变/阴影）",
    )
    parser.add_argument(
        "--val-threshold",
        type=float,
        default=70.0,
        help="明度容忍阈值（允许渐变/阴影）",
    )
    parser.add_argument("--min-region-area", type=int, default=16, help="最小区域像素数")
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


def quantize_colors_kmeans(image_rgb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """颜色量化/聚类：在 Lab 空间做 k-means。"""

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    num_samples = int(pixels.shape[0])
    if k < 1:
        raise ValueError(f"--num-clusters 必须 >= 1，当前值: {k}")
    effective_k = min(k, num_samples)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _compactness, labels, centers = cv2.kmeans(
        data=pixels,
        K=effective_k,
        bestLabels=None,
        criteria=criteria,
        attempts=5,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.reshape(image_rgb.shape[:2])
    centers_lab_u8 = np.clip(centers, 0, 255).astype(np.uint8)[None, :, :]
    centers_rgb = cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2RGB)[0]
    quantized = centers_rgb[labels]
    return quantized, labels


def circular_hue_distance(h1: float, h2: float) -> float:
    diff = abs(h1 - h2)
    return min(diff, 180.0 - diff)


def merge_similar_colors(
    labels: np.ndarray,
    palette_rgb: np.ndarray,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """按色相优先合并颜色，同时容忍明度/饱和度差异。"""

    palette_hsv = cv2.cvtColor(palette_rgb[None, :, :], cv2.COLOR_RGB2HSV)[0].astype(np.float32)

    groups: list[dict[str, np.ndarray | list[int]]] = []
    label_to_group = np.full(len(palette_rgb), -1, dtype=np.int32)

    for idx, hsv in enumerate(palette_hsv):
        assigned = False
        for gid, group in enumerate(groups):
            center_hsv = group["center_hsv"]
            hue_dist = circular_hue_distance(float(hsv[0]), float(center_hsv[0]))
            sat_dist = abs(float(hsv[1]) - float(center_hsv[1]))
            val_dist = abs(float(hsv[2]) - float(center_hsv[2]))

            if (
                hue_dist <= cfg.hue_threshold
                and sat_dist <= cfg.sat_threshold
                and val_dist <= cfg.val_threshold
            ):
                members = group["members"]
                assert isinstance(members, list)
                members.append(idx)
                member_hsv = palette_hsv[members]
                group["center_hsv"] = member_hsv.mean(axis=0)
                label_to_group[idx] = gid
                assigned = True
                break

        if not assigned:
            groups.append({"members": [idx], "center_hsv": hsv.copy()})
            label_to_group[idx] = len(groups) - 1

    merged_labels = label_to_group[labels]

    merged_palette = np.zeros((len(groups), 3), dtype=np.uint8)
    for gid, group in enumerate(groups):
        members = group["members"]
        assert isinstance(members, list)
        mean_rgb = palette_rgb[members].mean(axis=0)
        merged_palette[gid] = np.clip(mean_rgb, 0, 255).astype(np.uint8)

    return merged_labels, merged_palette


def build_region_instances(merged_labels: np.ndarray, min_region_area: int) -> np.ndarray:
    """对每种颜色做连通域分析，并分配全局 region ID。"""

    h, w = merged_labels.shape
    region_map = np.zeros((h, w), dtype=np.int32)
    next_id = 1

    for color_id in np.unique(merged_labels):
        mask = (merged_labels == color_id).astype(np.uint8)
        num, comp_ids, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for comp_idx in range(1, num):
            area = int(stats[comp_idx, cv2.CC_STAT_AREA])
            if area < min_region_area:
                continue
            region_map[comp_ids == comp_idx] = next_id
            next_id += 1

    return region_map


def labels_to_color(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[labels]


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
    axes[2].set_title("Merged by HSV")
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


def process_one_image(path: Path, output_dir: Path, cfg: PreprocessConfig, max_instances: int) -> None:
    image_rgb = read_color_image(path)

    # 1) 读取彩色图像
    # 2) 做颜色量化/聚类
    quantized_rgb, cluster_labels = quantize_colors_kmeans(image_rgb, cfg.num_clusters)

    # 将 kmeans 的离散 label 重新映射到连续 ID，方便后续 palette 索引。
    unique_cluster_ids = np.unique(cluster_labels)
    palette_rgb = np.array([quantized_rgb[cluster_labels == i][0] for i in unique_cluster_ids], dtype=np.uint8)
    remapped_cluster = np.zeros_like(cluster_labels)
    for new_id, old_id in enumerate(unique_cluster_ids):
        remapped_cluster[cluster_labels == old_id] = new_id

    # 3) 相同或相近颜色合并（优先看色相，对明度/纯度保留容忍）
    merged_labels, merged_palette = merge_similar_colors(remapped_cluster, palette_rgb, cfg)
    merged_rgb = labels_to_color(merged_labels, merged_palette)

    # 4) 对每种颜色做 connected components
    # 5) 每个连通块分配一个 region ID
    region_map = build_region_instances(merged_labels, cfg.min_region_area)

    # 将后缀编码进输出名前缀，避免同 stem 不同后缀时互相覆盖（如 scene.png / scene.jpg）。
    safe_suffix = path.suffix.lower().lstrip(".") or "nosuffix"
    stem_out = output_dir / f"{path.stem}__{safe_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 以 uint16 保存实例 ID（常见 8-bit 不足以承载大量区域 ID）。
    cv2.imwrite(
        str(stem_out.with_name(stem_out.name + "_region_id.png")),
        region_map.astype(np.uint16),
    )

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
        "num_colors_after_merge": int(len(np.unique(merged_labels))),
        "num_regions": int(region_map.max()),
        "thresholds": {
            "hue": cfg.hue_threshold,
            "saturation": cfg.sat_threshold,
            "value": cfg.val_threshold,
            "min_region_area": cfg.min_region_area,
        },
    }
    with stem_out.with_name(stem_out.name + "_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        num_clusters=args.num_clusters,
        hue_threshold=args.hue_threshold,
        sat_threshold=args.sat_threshold,
        val_threshold=args.val_threshold,
        min_region_area=args.min_region_area,
    )

    images = collect_images(args.input)
    if not images:
        raise ValueError(f"未在输入路径找到可处理图像: {args.input}")

    for img_path in images:
        process_one_image(
            path=img_path,
            output_dir=args.output,
            cfg=cfg,
            max_instances=args.max_vis_instances,
        )

    print(
        "完成处理:\n"
        f"- 图像数量: {len(images)}\n"
        f"- 输出目录: {args.output}\n"
        "- 产物: *_region_id.png, *_pipeline.png, *_instances.png, *_meta.json"
    )


if __name__ == "__main__":
    main()
