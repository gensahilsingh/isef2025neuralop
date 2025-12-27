r"""
fno.py
=====

3D Fourier Neural Operator (FNO) for mapping B -> J.

This is a baseline FNO block used inside the BPC-FNO pipeline. The physics
constraint is applied in training (train.py) via Biot–Savart consistency.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2)
        )

    def compl_mul3d(self, input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # enforce float32 for stability and consistent FFT behavior
        if x.dtype != torch.float32:
            x = x.float()

        batchsize, _, size_x, size_y, size_z = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        weight_complex = self.weight[..., 0] + 1j * self.weight[..., 1]
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            size_x,
            size_y,
            size_z // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], weight_complex
        )

        # passthrough higher freqs
        out_ft[:, :, : self.modes1, : self.modes2, self.modes3 :] = x_ft[
            :, :, : self.modes1, : self.modes2, self.modes3 :
        ]
        out_ft[:, :, : self.modes1, self.modes2 :, :] = x_ft[:, :, : self.modes1, self.modes2 :, :]
        out_ft[:, :, self.modes1 :, :, :] = x_ft[:, :, self.modes1 :, :, :]

        x = torch.fft.irfftn(out_ft, s=(size_x, size_y, size_z), dim=(-3, -2, -1))
        return x


class FNO3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: Tuple[int, int, int], width: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3 = modes
        self.width = width

        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()

        # (B,C,X,Y,Z) -> (B,X,Y,Z,C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        for conv, w in zip([self.conv0, self.conv1, self.conv2, self.conv3], [self.w0, self.w1, self.w2, self.w3]):
            x = F.gelu(conv(x) + w(x))

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


__all__ = ["SpectralConv3d", "FNO3d"]
