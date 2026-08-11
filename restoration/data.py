from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .io import image_files, read_grayscale


@dataclass(frozen=True)
class Pair:
    degraded: Path
    target: Path
    key: str


def _key(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace("\\", "/")


def discover_pairs(data_root: str | Path) -> tuple[list[Pair], list[str]]:
    root = Path(data_root)
    degraded_root = next((root / name for name in ("degraded", "NoisyLR") if (root / name).is_dir()), root / "degraded")
    target_root = next((root / name for name in ("ground_truth", "GT") if (root / name).is_dir()), root / "ground_truth")
    if not degraded_root.is_dir() or not target_root.is_dir():
        raise FileNotFoundError(f"Expected degraded/ground_truth/ or the official NoisyLR/GT/ folders under {root}")
    degraded = {_key(path, degraded_root): path for path in image_files(degraded_root)}
    target = {_key(path, target_root): path for path in image_files(target_root)}
    problems = [f"missing ground truth: {key}" for key in sorted(degraded.keys() - target.keys())]
    problems += [f"missing degraded image: {key}" for key in sorted(target.keys() - degraded.keys())]
    return [Pair(degraded[key], target[key], key) for key in sorted(degraded.keys() & target.keys())], problems


def robust_normalize(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize each CHW image using median/MAD without clipping noisy pixels."""
    flat = image.flatten(1)
    center_value = flat.median(dim=1).values
    center = center_value[:, None, None]
    mad = (flat - center_value[:, None]).abs().median(dim=1).values[:, None, None]
    scale = (1.4826 * mad).clamp_min(1e-3)
    return (image - center) / scale, center, scale


class PairedRestorationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, pairs: list[Pair], patch_size: int | None = 128, augment: bool = False, synthetic_probability: float = 0.35):
        self.pairs = pairs
        self.patch_size = patch_size
        self.augment = augment
        self.synthetic_probability = synthetic_probability

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _tensor(path: Path) -> torch.Tensor:
        return torch.from_numpy(read_grayscale(path)).unsqueeze(0)

    def _crop(self, low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if high.shape[-2] != low.shape[-2] * 2 or high.shape[-1] != low.shape[-1] * 2:
            raise ValueError(f"Expected a 2x pair, got degraded {tuple(low.shape)} and target {tuple(high.shape)}")
        if self.patch_size is None:
            return low, high
        patch = min(self.patch_size, low.shape[-2], low.shape[-1])
        top = random.randint(0, low.shape[-2] - patch)
        left = random.randint(0, low.shape[-1] - patch)
        return low[:, top:top + patch, left:left + patch], high[:, 2 * top:2 * (top + patch), 2 * left:2 * (left + patch)]

    @staticmethod
    def _synthetic_degrade(high: torch.Tensor) -> torch.Tensor:
        # Blur is deliberately applied before downsampling: this mirrors loss of
        # high-frequency inspection detail instead of merely softening the low-res input.
        sigma = random.uniform(0.1, 1.2)
        radius = 2
        axis = torch.arange(-radius, radius + 1, dtype=high.dtype, device=high.device)
        kernel_1d = torch.exp(-0.5 * (axis / sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = torch.outer(kernel_1d, kernel_1d)[None, None]
        blurred = F.conv2d(high.unsqueeze(0), kernel, padding=radius).squeeze(0)
        low = F.interpolate(blurred.unsqueeze(0), scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True).squeeze(0)
        speckle_sigma = random.uniform(0.02, 0.18)
        gaussian_sigma = random.uniform(0.0, 0.06)
        return low * (1.0 + torch.randn_like(low) * speckle_sigma) + torch.randn_like(low) * gaussian_sigma

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pair = self.pairs[index]
        low, high = self._tensor(pair.degraded), self._tensor(pair.target)
        low, high = self._crop(low, high)
        if self.augment:
            if random.random() < 0.5:
                low, high = low.flip(-1), high.flip(-1)
            if random.random() < 0.5:
                low, high = low.flip(-2), high.flip(-2)
            if random.random() < self.synthetic_probability:
                low = self._synthetic_degrade(high)
        low_norm, center, scale = robust_normalize(low)
        high_norm = (high - center) / scale
        return low_norm, high_norm, pair.key
