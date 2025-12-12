# mig_forward.py
import torch

# ============================================================
#  SENSOR GRID
# ============================================================

def make_sensor_grid(Nsx=16, Nsy=16, device="cpu"):
    """
    Returns a (Nsx*Nsy, 3) grid of sensor coordinates.
    Grid spans ~8cm × 8cm square at z = 0.06 m.
    """
    xs = torch.linspace(-0.04, 0.04, Nsx, device=device)
    ys = torch.linspace(-0.04, 0.04, Nsy, device=device)
    zs = torch.tensor([0.06], device=device)

    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    return coords  # (256,3)


# ============================================================
#  BIOT–SAVART FORWARD MODEL
# ============================================================

def forward_biot_savart_sensors(J, sensor_coords):
    """
    Compute B-field at sensors.

    J:
        (3,Nx,Ny,Nz) or (B,3,Nx,Ny,Nz)
    sensor_coords:
        (S,3)

    RETURNS:
        (B, S, 3)
    """

    # -----------------------------
    # (1) enforce batch
    # -----------------------------
    if J.dim() == 4:
        J = J.unsqueeze(0)

    Bsz, C, Nx, Ny, Nz = J.shape
    assert C == 3

    S = sensor_coords.shape[0]
    device = J.device

    # -----------------------------
    # (2) Flatten J → (B,P,3)
    # -----------------------------
    P = Nx * Ny * Nz
    J_flat = J.reshape(Bsz, 3, P).permute(0, 2, 1)
    assert J_flat.shape == (Bsz, P, 3)

    # -----------------------------
    # (3) voxel coords → (1,1,P,3)
    # -----------------------------
    xs = torch.linspace(-0.015, 0.015, Nx, device=device)
    ys = torch.linspace(-0.015, 0.015, Ny, device=device)
    zs = torch.linspace(0.00,  0.03, Nz, device=device)

    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    vox = torch.stack([X, Y, Z], dim=-1).reshape(P, 3)
    vox = vox.unsqueeze(0).unsqueeze(0)  # (1,1,P,3)
    assert vox.shape == (1,1,P,3)

    # -----------------------------
    # (4) sensor coords → (1,S,1,3)
    # -----------------------------
    sensors = sensor_coords.to(device).reshape(1, S, 1, 3)
    assert sensors.shape == (1,S,1,3)

    # -----------------------------
    # (5) r = sensor - voxel
    # -----------------------------
    r = sensors - vox          # (1,S,P,3)
    dist = torch.norm(r, dim=-1) + 1e-6   # (1,S,P)
    r_hat = r / dist.unsqueeze(-1)        # (1,S,P,3)

    # expand r_hat to B batch
    r_hat = r_hat.expand(Bsz, S, P, 3)
    assert r_hat.shape == (Bsz,S,P,3)

    # -----------------------------
    # (6) expand current
    # -----------------------------
    J_exp = J_flat.unsqueeze(1).expand(Bsz, S, P, 3)
    assert J_exp.shape == (Bsz,S,P,3)

    # -----------------------------
    # (7) cross product
    # -----------------------------
    cross = torch.cross(J_exp, r_hat, dim=-1)
    assert cross.shape == (Bsz,S,P,3)

    # -----------------------------
    # (8) denominator → expand to match cross
    # -----------------------------
    # dist: (1, S, P)
    denom = (dist.unsqueeze(-1) ** 2)    # (1, S, P, 1)
    denom = denom.expand(Bsz, S, P, 1)   # (B, S, P, 1)

    assert denom.shape == (Bsz,S,P,1)

    contrib = cross / denom
    assert contrib.shape == (Bsz,S,P,3)

    # -----------------------------
    # (9) integrate over voxels
    # -----------------------------
    B_sensors = contrib.sum(dim=2)   # (B,S,3)
    assert B_sensors.shape == (Bsz,S,3)
    dV = (2*0.015 / Nx) * (2*0.015 / Ny) * (0.03 / Nz)  # or your dx,dy,dz exactly

    B_sensors = contrib.sum(dim=2) * dV

    return B_sensors
