from __future__ import annotations

import torch


def split_disconnected_regions(label_map: torch.Tensor) -> torch.Tensor:
    """将同一 label 的不连通块拆分并重新编号。"""
    if label_map.ndim != 3:
        raise ValueError("label_map should be [N, H, W].")

    out = torch.zeros_like(label_map)
    n, h, w = out.shape
    pixel_ids = torch.arange(1, h * w + 1, device=label_map.device, dtype=torch.int64).view(h, w)

    for b in range(n):
        labels = label_map[b]
        fg = labels > 0
        roots = torch.where(fg, pixel_ids, torch.zeros_like(pixel_ids))

        while True:
            up_roots = torch.roll(roots, shifts=1, dims=0)
            up_labels = torch.roll(labels, shifts=1, dims=0)
            up_roots[0, :] = 0
            up_labels[0, :] = 0

            down_roots = torch.roll(roots, shifts=-1, dims=0)
            down_labels = torch.roll(labels, shifts=-1, dims=0)
            down_roots[-1, :] = 0
            down_labels[-1, :] = 0

            left_roots = torch.roll(roots, shifts=1, dims=1)
            left_labels = torch.roll(labels, shifts=1, dims=1)
            left_roots[:, 0] = 0
            left_labels[:, 0] = 0

            right_roots = torch.roll(roots, shifts=-1, dims=1)
            right_labels = torch.roll(labels, shifts=-1, dims=1)
            right_roots[:, -1] = 0
            right_labels[:, -1] = 0

            nbr_roots = torch.stack((up_roots, down_roots, left_roots, right_roots), dim=0)
            nbr_labels = torch.stack((up_labels, down_labels, left_labels, right_labels), dim=0)
            same_label = nbr_labels == labels.unsqueeze(0)
            candidates = torch.where(same_label, nbr_roots, torch.zeros_like(nbr_roots))
            candidates = torch.where(candidates > 0, candidates, torch.full_like(candidates, torch.iinfo(torch.int64).max))
            min_candidate = candidates.amin(dim=0)
            min_candidate = torch.where(min_candidate == torch.iinfo(torch.int64).max, roots, min_candidate)
            new_roots = torch.where(fg, torch.minimum(roots, min_candidate), torch.zeros_like(roots))
            if not bool((new_roots != roots).any()):
                break
            roots = new_roots

        comp_ids = torch.unique(roots[roots > 0], sorted=True)
        if comp_ids.numel() == 0:
            continue
        mapped = torch.searchsorted(comp_ids, roots)
        out[b] = torch.where(roots > 0, mapped + 1, torch.zeros_like(mapped))
    return out


def filter_small_regions(label_map: torch.Tensor, min_area: int = 16) -> torch.Tensor:
    """移除面积小于阈值的小区域（置零）。"""
    if min_area <= 0:
        return label_map

    out = label_map.clone()
    for b in range(out.shape[0]):
        labels = out[b]
        max_id = int(labels.max().item())
        if max_id <= 0:
            continue
        counts = torch.bincount(labels.view(-1), minlength=max_id + 1)
        keep = counts >= min_area
        keep[0] = False
        out[b] = torch.where(keep[labels], labels, torch.zeros_like(labels))
    return out


def relabel_sequential(label_map: torch.Tensor) -> torch.Tensor:
    """将区域 id 重新映射到 1..K，便于下游处理。"""
    out = torch.zeros_like(label_map)
    for b in range(label_map.shape[0]):
        labels = label_map[b]
        ids = torch.unique(labels[labels > 0], sorted=True)
        if ids.numel() == 0:
            continue
        mapped = torch.searchsorted(ids, labels)
        out[b] = torch.where(labels > 0, mapped + 1, torch.zeros_like(labels))
    return out


def fill_unassigned_pixels(label_map: torch.Tensor, interior_bin: torch.Tensor, max_iter: int = 64) -> torch.Tensor:
    """对 interior 内仍为 0 的像素，使用邻域多数投票进行补全。"""
    if label_map.ndim != 3:
        raise ValueError("label_map should be [N, H, W].")
    if interior_bin.ndim != 4 or interior_bin.shape[1] != 1:
        raise ValueError("interior_bin should be [N, 1, H, W].")

    out = label_map.clone()
    n = out.shape[0]

    for b in range(n):
        valid = interior_bin[b, 0] > 0
        labels = out[b]
        for _ in range(max_iter):
            unlabeled = (labels == 0) & valid
            if not bool(unlabeled.any()):
                break

            up = torch.roll(labels, shifts=1, dims=0)
            up[0, :] = 0
            down = torch.roll(labels, shifts=-1, dims=0)
            down[-1, :] = 0
            left = torch.roll(labels, shifts=1, dims=1)
            left[:, 0] = 0
            right = torch.roll(labels, shifts=-1, dims=1)
            right[:, -1] = 0
            neighbors = torch.stack((up, down, left, right), dim=0)

            positive = neighbors > 0
            vote_counts = torch.zeros_like(neighbors)
            for k in range(4):
                vote_counts[k] = ((neighbors == neighbors[k : k + 1]) & positive).sum(dim=0)
            vote_counts = torch.where(positive, vote_counts, torch.zeros_like(vote_counts))

            best_idx = vote_counts.argmax(dim=0, keepdim=True)
            best_label = torch.gather(neighbors, 0, best_idx).squeeze(0)
            changed = unlabeled & (best_label > 0)
            if not bool(changed.any()):
                break
            labels = torch.where(changed, best_label, labels)
        out[b] = labels

    return out
