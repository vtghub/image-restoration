from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel_x = prediction.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    def gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.conv2d(image, kernel_x, padding=1), F.conv2d(image, kernel_y, padding=1)
    px, py = gradients(prediction)
    tx, ty = gradients(target)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def restoration_loss(prediction: torch.Tensor, target: torch.Tensor, gradient_weight: float = 0.10) -> torch.Tensor:
    return charbonnier(prediction, target) + gradient_weight * gradient_loss(prediction, target)
