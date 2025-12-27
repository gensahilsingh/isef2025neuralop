r"""
biot_savart.py
================

Discretized Biot–Savart law for volume current distributions.

Implements:
    B(r) = (mu0 / 4pi) ∫_V  J(r') × (r - r') / ||r - r'||^3  dV'

Notes
-----
- This is a brute-force O(N^2) implementation (N = nx*ny*nz), intended
  for small grids. It is differentiable and works on CPU or CUDA.
"""

from __future__ import annotations

from typing import Tuple
import torch


def compute_biot_savart(
    J: torch.Tensor,
    dx: float = 1.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Compute magnetic flux density B from a current density J on a regular grid.

    Parameters
    ----------
    J : torch.Tensor
        Shape (nx, ny, nz, 3) or (batch, nx, ny, nz, 3)
        Last dimension corresponds to (Jx, Jy, Jz).

    dx : float
        Voxel spacing. dV = dx^3.

    eps : float
        Small constant added to distance to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Magnetic field B with same shape as J.
    """
    if J.dim() not in (4, 5):
        raise ValueError(f"J must have dim 4 or 5, got {J.dim()} with shape {tuple(J.shape)}")

    input_had_batch = (J.dim() == 5)
    if not input_had_batch:
        J = J.unsqueeze(0)  # (1, nx, ny, nz, 3)

    # enforce float32 for stable CUDA kernels
    if J.dtype != torch.float32:
        J = J.float()

    batch, nx, ny, nz, _ = J.shape
    device = J.device

    # coordinate grid centered at 0
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(nx, dtype=torch.float32, device=device),
            torch.arange(ny, dtype=torch.float32, device=device),
            torch.arange(nz, dtype=torch.float32, device=device),
            indexing="ij",
        ),
        dim=-1,
    )  # (nx, ny, nz, 3)

    center = torch.tensor([nx - 1, ny - 1, nz - 1], dtype=torch.float32, device=device) / 2.0
    coords = (coords - center) * dx
    coords_flat = coords.reshape(-1, 3)  # (N,3)
    N = coords_flat.shape[0]

    # constants
    mu0_over_4pi = 1e-7
    factor = mu0_over_4pi * (dx ** 3)

    # flatten J: (batch, N, 3)
    J_flat = J.reshape(batch, N, 3)

    # precompute geometry once (independent of batch)
    r_diff = coords_flat.unsqueeze(1) - coords_flat.unsqueeze(0)  # (N,N,3) where diff[i,j]=ri-rj
    dist = torch.linalg.norm(r_diff, dim=2) + eps  # (N,N)
    inv_dist3 = 1.0 / (dist ** 3)  # (N,N)

    B_flat = torch.zeros((batch, N, 3), dtype=torch.float32, device=device)

    for b in range(batch):
        # Jb (N,3) -> broadcast to (N,N,3) as sources across j
        # we want for each observation i: sum_j J(j) x (ri-rj)/||ri-rj||^3
        Jb = J_flat[b]  # (N,3)
        J_src = Jb.unsqueeze(0).expand(N, N, 3)  # (N,N,3)
        cross = torch.cross(J_src, r_diff, dim=2)  # (N,N,3)
        contrib = cross * inv_dist3.unsqueeze(2)  # (N,N,3)
        B_flat[b] = factor * contrib.sum(dim=1)  # sum over sources j

    B = B_flat.reshape(batch, nx, ny, nz, 3)

    if not input_had_batch:
        B = B.squeeze(0)  # back to (nx,ny,nz,3)

    return B


__all__ = ["compute_biot_savart"]
