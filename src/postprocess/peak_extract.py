from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """生成 2D 高斯核（用于 heatmap 平滑）。"""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = torch.outer(gauss_1d, gauss_1d)
    return kernel_2d / kernel_2d.sum()


def smooth_heatmap(seed_heatmap: torch.Tensor, kernel_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """对 Seed Heatmap 做高斯平滑。

    Args:
        seed_heatmap: [N, 1, H, W] 的预测 heatmap。
    """
    if seed_heatmap.ndim != 4 or seed_heatmap.shape[1] != 1:
        raise ValueError("seed_heatmap should be [N, 1, H, W].")

    kernel = gaussian_kernel2d(kernel_size, sigma, seed_heatmap.device, seed_heatmap.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    return F.conv2d(seed_heatmap, kernel, padding=padding)


def extract_peak_mask(
    seed_heatmap: torch.Tensor,
    interior_bin: torch.Tensor,
    peak_threshold: float = 0.35,
    min_distance: int = 5,
) -> torch.Tensor:
    """从 heatmap 提取局部峰值，作为 marker 候选。

    Returns:
        [N, 1, H, W] bool tensor，True 表示峰值像素。
    """
    if min_distance < 1:
        raise ValueError("min_distance must be >= 1")

    if seed_heatmap.shape != interior_bin.shape:
        raise ValueError("seed_heatmap and interior_bin must share the same shape.")

    window = 2 * min_distance + 1
    local_max = F.max_pool2d(seed_heatmap, kernel_size=window, stride=1, padding=min_distance)

    peak_mask = (seed_heatmap >= peak_threshold) & (seed_heatmap >= local_max)
    peak_mask = peak_mask & (interior_bin > 0)
    return peak_mask


def peak_mask_to_markers(peak_mask: torch.Tensor) -> torch.Tensor:
    """将峰值 mask 转换为 marker 图。

    简化策略：每个峰值像素即一个 marker，按扫描顺序分配递增 id。

    Returns:
        [N, H, W] int64 marker map（0 表示背景）。
    """
    if peak_mask.ndim != 4 or peak_mask.shape[1] != 1:
        raise ValueError("peak_mask should be [N, 1, H, W].")

    n, _, h, w = peak_mask.shape
    markers = torch.zeros((n, h, w), dtype=torch.int64, device=peak_mask.device)

    for b in range(n):
        coords = torch.nonzero(peak_mask[b, 0], as_tuple=False)
        if coords.numel() == 0:
            continue
        marker_ids = torch.arange(1, coords.shape[0] + 1, device=peak_mask.device, dtype=torch.int64)
        markers[b, coords[:, 0], coords[:, 1]] = marker_ids

    return markers
