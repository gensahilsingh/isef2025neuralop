# mig_geometry.py

import torch

def make_volume_coords(Nx, Ny, Nz, dx, dy, dz, device="cpu"):
    """
    Create 3D voxel center coordinates for the heart volume.
    Returns:
      X, Y, Z: each (Nx, Ny, Nz) tensors with coordinates in meters.
    """
    xs = (torch.arange(Nx, device=device) + 0.5) * dx
    ys = (torch.arange(Ny, device=device) + 0.5) * dy
    zs = (torch.arange(Nz, device=device) + 0.5) * dz
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    return X, Y, Z


def make_sensor_plane(Nsx, Nsy, X, Y, Z, sensor_offset=0.01, device="cpu"):
    """
    Create a 2D sensor grid (SQUID array) above the heart volume.
    - Nsx, Nsy: number of sensors in x and y.
    - sensor_offset: distance above the top of the volume (meters).
    Returns:
      coords_sensors: (Ns, 3) tensor of sensor coordinates.
      H, W: Nsx, Nsy (for reshaping later).
    """
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_top = Z.max() + sensor_offset

    xs = torch.linspace(x_min, x_max, Nsx, device=device)
    ys = torch.linspace(y_min, y_max, Nsy, device=device)
    Xs, Ys = torch.meshgrid(xs, ys, indexing="ij")
    Zs = torch.full_like(Xs, z_top)

    coords = torch.stack([Xs, Ys, Zs], dim=-1)  # (Nsx, Nsy, 3)
    coords_flat = coords.view(-1, 3)           # (Ns, 3)
    return coords_flat, Nsx, Nsy
