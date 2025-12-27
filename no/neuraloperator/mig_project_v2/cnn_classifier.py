r"""
cnn_classifier.py
=================

3D CNN classifier for disease state from reconstructed current fields.

Key hardening:
- always aligns x to model device
- forces float32
- forces contiguous (prevents slow_conv3d_forward CUDA crash)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeartDiseaseClassifier3D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32, dropout_prob: float = 0.2) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=float(dropout_prob))
        self.fc = nn.Linear(base_channels * 8, self.num_classes)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # move to model device, force float32, force contiguous
        dev = next(self.parameters()).device
        x = x.to(dev)
        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prep(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    @torch.no_grad()
    def predict_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self._prep(x)
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)

        magnitude = torch.sqrt((x ** 2).sum(dim=1) + 1e-12)  # (B,D,H,W)
        mean_mag = magnitude.mean(dim=[1, 2, 3])
        std_mag = magnitude.std(dim=[1, 2, 3])

        flat_mag = magnitude.reshape(magnitude.shape[0], -1)
        max_idx = torch.argmax(flat_mag, dim=1)

        nx, ny, nz = magnitude.shape[1:]
        z_idx = max_idx % nz
        y_idx = (max_idx // nz) % ny
        x_idx = (max_idx // (ny * nz)) % nx
        max_coord = torch.stack((x_idx, y_idx, z_idx), dim=1)

        features: Dict[str, torch.Tensor] = {
            "mean_magnitude": mean_mag.detach().cpu(),
            "std_magnitude": std_mag.detach().cpu(),
            "max_magnitude_coord": max_coord.detach().cpu(),
        }
        return probs.detach().cpu(), features


__all__ = ["HeartDiseaseClassifier3D"]
