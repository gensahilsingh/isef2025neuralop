# mig_patterns.py

import torch

def plane_wave_current(X, Y, Z, direction, center_plane, thickness, amplitude):
    """
    Simulate a plane wave of activation propagating along 'direction'.
    direction: (3,) tensor (not necessarily unit, we normalize)
    center_plane: scalar position along direction where the wavefront is centered.
    thickness: spatial thickness of the wavefront (meters).
    amplitude: scale factor for current magnitude.
    """
    n = direction / (torch.norm(direction) + 1e-8)
    proj = n[0] * X + n[1] * Y + n[2] * Z  # projection onto direction
    g = torch.exp(-0.5 * ((proj - center_plane) / thickness) ** 2)

    Jx = amplitude * n[0] * g
    Jy = amplitude * n[1] * g
    Jz = amplitude * n[2] * g

    return torch.stack([Jx, Jy, Jz], dim=0)


def focal_current(X, Y, Z, center, sigma_space, direction, amplitude):
    """
    Simulate a focal activation region that decays radially.
    center: (3,) tensor [cx, cy, cz]
    sigma_space: spatial spread (meters)
    direction: (3,) tensor for main current direction
    amplitude: scale factor
    """
    cx, cy, cz = center
    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    g = torch.exp(-0.5 * r2 / (sigma_space ** 2))

    n = direction / (torch.norm(direction) + 1e-8)
    Jx = amplitude * n[0] * g
    Jy = amplitude * n[1] * g
    Jz = amplitude * n[2] * g

    return torch.stack([Jx, Jy, Jz], dim=0)


def reentry_ring(X, Y, Z, center, radius, thickness, plane_z, amplitude):
    """
    Simulate a reentry loop: current circulating in a ring in the XY plane near plane_z.
    center: (3,) tensor [cx, cy, cz]
    radius: ring radius (meters)
    thickness: radial + z thickness (meters)
    plane_z: z position of ring center (meters)
    amplitude: scale factor
    """
    cx, cy, _ = center

    # z envelope
    g_z = torch.exp(-0.5 * ((Z - plane_z) / thickness) ** 2)

    # radial envelope around the ring
    r_xy = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    g_r = torch.exp(-0.5 * ((r_xy - radius) / thickness) ** 2)

    g = g_z * g_r

    # tangential direction around ring
    tx = -(Y - cy)
    ty = (X - cx)
    norm = torch.sqrt(tx ** 2 + ty ** 2) + 1e-8
    tx = tx / norm
    ty = ty / norm

    Jx = amplitude * tx * g
    Jy = amplitude * ty * g
    Jz = torch.zeros_like(Jx)

    return torch.stack([Jx, Jy, Jz], dim=0)


def conduction_block(X, Y, Z, base_pattern, block_center, block_size):
    """
    Apply a conduction block by zeroing currents in a cuboid region.
    base_pattern: (3,Nx,Ny,Nz) current field to modify
    block_center: (3,) tensor [cx, cy, cz]
    block_size: (3,) tensor [sx, sy, sz] half-sizes of block (meters)
    """
    cx, cy, cz = block_center
    sx, sy, sz = block_size

    mask_x = (X > (cx - sx)) & (X < (cx + sx))
    mask_y = (Y > (cy - sy)) & (Y < (cy + sy))
    mask_z = (Z > (cz - sz)) & (Z < (cz + sz))

    block_mask = mask_x & mask_y & mask_z  # (Nx,Ny,Nz)
    blocked = base_pattern.clone()
    blocked[:, block_mask] = 0.0
    return blocked


def sample_current_pattern(X, Y, Z, device="cpu"):
    """
    Randomly sample a realistic-ish current pattern by mixing plane, focal, reentry, and block.
    Returns:
      J_total: (3,Nx,Ny,Nz)
      disease_label: int (0=normal, 1=focal arrhythmia, 2=reentry, 3=block)
    """
    X = X.to(device)
    Y = Y.to(device)
    Z = Z.to(device)

    J_total = torch.zeros(3, *X.shape, device=device)

    # random choice of base phenotype
    r = torch.rand(1).item()
    if r < 0.25:
        # normal conduction plane wave (label 0)
        direction = torch.randn(3, device=device)
        center_plane = torch.rand(1, device=device) * X.max()
        thickness = 0.005 + 0.01 * torch.rand(1, device=device)
        amp = 1.0 * (0.5 + torch.rand(1, device=device))
        J_total = plane_wave_current(X, Y, Z, direction, center_plane, thickness, amp)
        label = 0
    elif r < 0.5:
        # focal arrhythmia (label 1)
        center = torch.stack([
            torch.rand(1, device=device) * X.max(),
            torch.rand(1, device=device) * Y.max(),
            torch.rand(1, device=device) * Z.max()
        ]).flatten()
        sigma_space = 0.005 + 0.01 * torch.rand(1, device=device)
        direction = torch.randn(3, device=device)
        amp = 1.0 * (0.5 + torch.rand(1, device=device))
        J_total = focal_current(X, Y, Z, center, sigma_space, direction, amp)
        label = 1
    elif r < 0.75:
        # reentry ring (label 2)
        center = torch.stack([
            0.5 * X.max(),
            0.5 * Y.max(),
            0.5 * Z.max()
        ]).flatten()
        radius = 0.01 + 0.01 * torch.rand(1, device=device)
        thickness = 0.003 + 0.007 * torch.rand(1, device=device)
        plane_z = center[2]
        amp = 1.0 * (0.5 + torch.rand(1, device=device))
        J_total = reentry_ring(X, Y, Z, center, radius, thickness, plane_z, amp)
        label = 2
    else:
        # normal plane wave with a conduction block (label 3)
        direction = torch.randn(3, device=device)
        center_plane = torch.rand(1, device=device) * X.max()
        thickness = 0.005 + 0.01 * torch.rand(1, device=device)
        amp = 1.0 * (0.5 + torch.rand(1, device=device))
        J_base = plane_wave_current(X, Y, Z, direction, center_plane, thickness, amp)

        block_center = torch.stack([
            0.5 * X.max(),
            0.5 * Y.max(),
            0.5 * Z.max()
        ]).flatten()
        block_size = torch.tensor([0.01, 0.01, 0.01], device=device)
        J_total = conduction_block(X, Y, Z, J_base, block_center, block_size)
        label = 3

    # normalize magnitude so training is stable
    max_abs = J_total.abs().max() + 1e-8
    J_total = J_total / max_abs
    return J_total, label
