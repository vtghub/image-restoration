from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.project = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.project(F.gelu(self.depthwise(F.gelu(self.expand(self.norm(x))))))
        return x + residual


class JointRestorationNet(nn.Module):
    """Compact residual network with a 2x pixel-shuffle restoration head."""
    def __init__(self, width: int = 48, blocks: int = 10):
        super().__init__()
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.fusion = nn.Conv2d(width, width, 3, padding=1)
        self.upscale = nn.Sequential(nn.Conv2d(width, width * 4, 3, padding=1), nn.PixelShuffle(2), nn.GELU(), nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, low: torch.Tensor) -> torch.Tensor:
        features = self.stem(low)
        features = self.fusion(self.body(features)) + features
        residual = self.upscale(features)
        base = F.interpolate(low, scale_factor=2, mode="bicubic", align_corners=False)
        return base + residual

