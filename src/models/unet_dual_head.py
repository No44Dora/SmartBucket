from __future__ import annotations

import torch
from torch import nn

from .blocks import DoubleConv, Down, Up


class UNetDualHead(nn.Module):
    """双头 U-Net（固定通道配置）。

    目标结构：
    - Input: 1 x 512 x 512
    - Encoder: 32 -> 64 -> 128 -> 256
    - Bottleneck: 512
    - Decoder: 256 -> 128 -> 64 -> 32
    - Outputs: interior(1ch), seed(1ch)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, 32)   # 512x512
        self.down1 = Down(32, 64)                # 256x256
        self.down2 = Down(64, 128)               # 128x128
        self.down3 = Down(128, 256)              # 64x64
        self.down4 = Down(256, 512)              # 32x32 (bottleneck)

        # Decoder
        self.up1 = Up(512, 256)                  # 64x64
        self.up2 = Up(256, 128)                  # 128x128
        self.up3 = Up(128, 64)                   # 256x256
        self.up4 = Up(64, 32)                    # 512x512

        # Output Heads
        self.interior_head = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.seed_head = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)

        interior_pred = self.interior_head(y)
        seed_pred = self.seed_head(y)
        return interior_pred, seed_pred
