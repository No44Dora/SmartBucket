from __future__ import annotations

from collections import defaultdict

import torch


def filter_small_regions(label_map: torch.Tensor, min_area: int = 16) -> torch.Tensor:
    """移除面积小于阈值的小区域（置零）。"""
    if min_area <= 0:
        return label_map

    out = label_map.clone()
    for b in range(out.shape[0]):
        labels = out[b]
        ids, counts = torch.unique(labels, return_counts=True)
        for region_id, area in zip(ids.tolist(), counts.tolist()):
            if region_id == 0:
                continue
            if area < min_area:
                labels[labels == region_id] = 0
    return out


def relabel_sequential(label_map: torch.Tensor) -> torch.Tensor:
    """将区域 id 重新映射到 1..K，便于下游处理。"""
    out = label_map.clone()
    for b in range(out.shape[0]):
        labels = out[b]
        ids = torch.unique(labels)
        ids = ids[ids > 0]
        new_labels = torch.zeros_like(labels)
        for new_id, old_id in enumerate(ids.tolist(), start=1):
            new_labels[labels == old_id] = new_id
        out[b] = new_labels
    return out


def fill_unassigned_pixels(label_map: torch.Tensor, interior_bin: torch.Tensor, max_iter: int = 64) -> torch.Tensor:
    """对 interior 内仍为 0 的像素，使用邻域多数投票进行补全。"""
    if label_map.ndim != 3:
        raise ValueError("label_map should be [N, H, W].")
    if interior_bin.ndim != 4 or interior_bin.shape[1] != 1:
        raise ValueError("interior_bin should be [N, 1, H, W].")

    out = label_map.clone()
    n, h, w = out.shape

    for b in range(n):
        valid = interior_bin[b, 0] > 0
        for _ in range(max_iter):
            changed = False
            unassigned = torch.nonzero((out[b] == 0) & valid, as_tuple=False)
            if unassigned.numel() == 0:
                break

            for y, x in unassigned.tolist():
                votes: dict[int, int] = defaultdict(int)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    lbl = int(out[b, ny, nx].item())
                    if lbl > 0:
                        votes[lbl] += 1

                if votes:
                    best_label = max(votes.items(), key=lambda item: item[1])[0]
                    out[b, y, x] = best_label
                    changed = True

            if not changed:
                break

    return out
