# mig_forward.py

import torch

MU0 = 4e-7 * torch.pi  # magnetic constant μ0

def flatten_volume_coords(X, Y, Z):
    """
    X,Y,Z: (Nx,Ny,Nz)
    Returns coords: (Nv,3) where Nv = Nx*Ny*Nz
    """
    coords = torch.stack([X, Y, Z], dim=-1)  # (Nx,Ny,Nz,3)
    return coords.view(-1, 3)


def forward_biot_savart_sensors(J, X, Y, Z, sensor_coords, dx, dy, dz):
    """
    Compute B at each sensor due to current field J.

    J: (3,Nx,Ny,Nz)
    X,Y,Z: (Nx,Ny,Nz)
    sensor_coords: (Ns,3)
    dx,dy,dz: voxel sizes (meters)
    Returns:
      B_sensors: (Ns,3)
    """
    device = J.device
    dV = dx * dy * dz

    # flatten
    voxel_coords = flatten_volume_coords(X, Y, Z).to(device)   # (Nv,3)
    J_flat = J.view(3, -1).transpose(0, 1)                     # (Nv,3)
    sensor_coords = sensor_coords.to(device)                   # (Ns,3)

    Nv = voxel_coords.shape[0]
    Ns = sensor_coords.shape[0]

    # expand dims: rs (Ns,1,3), rj (1,Nv,3)
    rs = sensor_coords.unsqueeze(1)            # (Ns,1,3)
    rj = voxel_coords.unsqueeze(0)            # (1,Nv,3)
    R = rs - rj                                # (Ns,Nv,3)
    R_norm = torch.norm(R, dim=-1) + 1e-9     # (Ns,Nv)

    # J x R
    Jv = J_flat.unsqueeze(0).expand(Ns, Nv, 3)  # (Ns,Nv,3)
    JxR = torch.cross(Jv, R, dim=-1)            # (Ns,Nv,3)

    factor = MU0 / (4 * torch.pi)
    coeff = factor * dV / (R_norm ** 3).unsqueeze(-1)   # (Ns,Nv,1)
    contrib = coeff * JxR                               # (Ns,Nv,3)

    B = contrib.sum(dim=1)     # (Ns,3)
    return B
