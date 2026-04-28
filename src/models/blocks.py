from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """U-Net基础卷积块：Conv-BN-ReLU × 2。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 第一层 3x3 卷积，保持空间分辨率不变
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二层 3x3 卷积，进一步提取局部纹理
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """下采样模块：MaxPool 降采样后接 DoubleConv。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """上采样模块：反卷积上采样 + skip concat + DoubleConv。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 先将通道减半并上采样到 2x 空间尺寸
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # 当输入尺寸不是 2^n 时，可能出现1像素级对齐误差，这里做padding对齐
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # 通道维拼接 encoder 特征（skip connection）
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
