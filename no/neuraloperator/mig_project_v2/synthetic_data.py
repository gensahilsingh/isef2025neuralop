r"""
synthetic_data.py
=================

Synthetic data for magnetoionography / magnetocardiography style inversion.

Generates:
- Intracellular-like current density J on a 3D voxel grid (nx,ny,nz,3)
- Corresponding magnetic field B via Biot–Savart (nx,ny,nz,3)
- Disease labels (int)

All returned tensors are float32. Dataset wrapper returns channel-first
(C, X, Y, Z) for models.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from biot_savart import compute_biot_savart


def _swirl_field(grid_size: int) -> torch.Tensor:
    """
    Baseline swirling field (normal conduction-like):
        Jx = -y, Jy = x, Jz = 0  (in centered coordinates)
    Returns (grid,grid,grid,3)
    """
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    )
    center = (grid_size - 1) / 2.0
    rel = coords - center
    x = rel[..., 0]
    y = rel[..., 1]
    z = rel[..., 2]

    Jx = -y
    Jy = x
    Jz = torch.zeros_like(z)

    mag = torch.sqrt(Jx ** 2 + Jy ** 2 + 1e-12)
    Jx = Jx / (mag + 1e-6)
    Jy = Jy / (mag + 1e-6)
    return torch.stack((Jx, Jy, Jz), dim=-1).float()


def generate_current(
    grid_size: int,
    disease: str = "normal",
    rng: random.Random | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Generate a synthetic current density field J: (grid,grid,grid,3)
    disease in {'normal','ischemia','arrhythmia','hypertrophy'}
    """
    if rng is None:
        rng = random

    J = _swirl_field(grid_size)

    if disease == "ischemia":
        center = torch.tensor(
            [
                rng.uniform(0.3, 0.7) * (grid_size - 1),
                rng.uniform(0.3, 0.7) * (grid_size - 1),
                rng.uniform(0.3, 0.7) * (grid_size - 1),
            ],
            dtype=torch.float32,
        )
        radius = rng.uniform(0.15, 0.25) * (grid_size - 1)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(grid_size, dtype=torch.float32),
                torch.arange(grid_size, dtype=torch.float32),
                torch.arange(grid_size, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        )
        dist = torch.linalg.norm(coords - center, dim=-1)
        mask = (dist <= radius).float()
        J = J * (1.0 - 0.8 * mask.unsqueeze(-1))  # reduce in region

    elif disease == "arrhythmia":
        noise = torch.randn_like(J) * 0.5
        J = J + noise
        mag = torch.sqrt((J ** 2).sum(dim=-1) + 1e-12)
        J = J / (mag.unsqueeze(-1) + 1e-6)

    elif disease == "hypertrophy":
        center = torch.tensor(
            [
                rng.uniform(0.3, 0.7) * (grid_size - 1),
                rng.uniform(0.3, 0.7) * (grid_size - 1),
                rng.uniform(0.3, 0.7) * (grid_size - 1),
            ],
            dtype=torch.float32,
        )
        radius = rng.uniform(0.2, 0.3) * (grid_size - 1)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(grid_size, dtype=torch.float32),
                torch.arange(grid_size, dtype=torch.float32),
                torch.arange(grid_size, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        )
        dist = torch.linalg.norm(coords - center, dim=-1)
        mask = (dist <= radius).float()
        J = J * (1.0 + 0.8 * mask.unsqueeze(-1))  # increase in region

    # scale
    J = (J * float(scale)).float()
    return J


def generate_dataset(
    dataset_size: int,
    grid_size: int,
    diseases: List[str] | None = None,
    noise_level: float = 0.01,
    dx: float = 1.0,
    seed: int | None = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    """
    Returns:
        currents: list of (grid,grid,grid,3)
        fields:   list of (grid,grid,grid,3)
        labels:   list of ints in [0, num_classes-1]
    """
    if diseases is None:
        diseases = ["normal", "ischemia", "arrhythmia", "hypertrophy"]

    rng = random.Random(seed)

    currents: List[torch.Tensor] = []
    fields: List[torch.Tensor] = []
    labels: List[int] = []

    for _ in range(dataset_size):
        disease = rng.choice(diseases)
        label = diseases.index(disease)

        J = generate_current(grid_size, disease=disease, rng=rng).float()

        # Biot–Savart returns (grid,grid,grid,3) for non-batched input (fixed in biot_savart.py)
        B = compute_biot_savart(J, dx=dx).float()

        # noise
        if noise_level > 0:
            B = B + torch.randn_like(B) * float(noise_level)

        currents.append(J)
        fields.append(B)
        labels.append(int(label))

    return currents, fields, labels


class HeartCurrentDataset(Dataset):
    """
    For FNO training:
        input  B_noisy: (C,X,Y,Z)
        target J_true:  (C,X,Y,Z)
    """

    def __init__(self, currents: List[torch.Tensor], fields: List[torch.Tensor]) -> None:
        if len(currents) != len(fields):
            raise ValueError("currents and fields must have same length")
        self.currents = currents
        self.fields = fields

    def __len__(self) -> int:
        return len(self.currents)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        J = self.currents[idx].float()  # (X,Y,Z,3)
        B = self.fields[idx].float()    # (X,Y,Z,3)

        # channel-first: (3,X,Y,Z)
        B_cf = B.permute(3, 0, 1, 2).contiguous()
        J_cf = J.permute(3, 0, 1, 2).contiguous()
        return B_cf, J_cf


__all__ = ["generate_current", "generate_dataset", "HeartCurrentDataset"]
