from __future__ import annotations

import torch


def watershed_from_markers(interior_bin: torch.Tensor, markers: torch.Tensor) -> torch.Tensor:
    """基于 interior mask 与 markers 的近似 watershed（多源 BFS）。

    说明：
    - 仅在 interior_bin > 0 的像素内传播标签；
    - 若不同标签在同一时刻竞争同一像素，则该像素保持 0（可视为分水岭边界）。

    Args:
        interior_bin: [N, 1, H, W]，二值 interior。
        markers: [N, H, W]，marker id（0 表示无 marker）。

    Returns:
        [N, H, W]，region label map。
    """
    if interior_bin.ndim != 4 or interior_bin.shape[1] != 1:
        raise ValueError("interior_bin should be [N, 1, H, W].")
    if markers.ndim != 3:
        raise ValueError("markers should be [N, H, W].")

    n, _, h, w = interior_bin.shape
    if markers.shape != (n, h, w):
        raise ValueError("markers shape should be [N, H, W] and match interior spatial shape.")

    outputs: list[torch.Tensor] = []

    for b in range(n):
        valid = interior_bin[b, 0] > 0
        labels = markers[b].to(dtype=torch.int64) * valid.to(torch.int64)

        while True:
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
            has_neighbor = positive.any(dim=0)

            sentinel = torch.full_like(neighbors, fill_value=torch.iinfo(torch.int64).max)
            min_label = torch.where(positive, neighbors, sentinel).amin(dim=0)
            max_label = torch.where(positive, neighbors, torch.zeros_like(neighbors)).amax(dim=0)

            unlabeled_valid = (labels == 0) & valid
            assignable = unlabeled_valid & has_neighbor & (min_label == max_label)
            if not bool(assignable.any()):
                break

            labels = torch.where(assignable, min_label, labels)

        outputs.append(labels)

    return torch.stack(outputs, dim=0)
