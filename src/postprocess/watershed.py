from __future__ import annotations

from collections import deque

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
        marker_map = markers[b].clone().to(torch.int64)

        labels = torch.zeros((h, w), dtype=torch.int64, device=markers.device)
        labels[marker_map > 0] = marker_map[marker_map > 0]

        q: deque[tuple[int, int]] = deque()
        marker_coords = torch.nonzero(marker_map > 0, as_tuple=False)
        for y, x in marker_coords.tolist():
            q.append((y, x))

        while q:
            y, x = q.popleft()
            src_label = int(labels[y, x].item())
            if src_label == 0:
                continue

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if not bool(valid[ny, nx]):
                    continue

                dst_label = int(labels[ny, nx].item())
                if dst_label == 0:
                    labels[ny, nx] = src_label
                    q.append((ny, nx))
                elif dst_label != src_label:
                    labels[ny, nx] = 0

        outputs.append(labels)

    return torch.stack(outputs, dim=0)
