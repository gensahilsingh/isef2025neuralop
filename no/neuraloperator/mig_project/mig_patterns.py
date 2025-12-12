# mig_patterns.py
import torch
import math

"""
Advanced synthetic current patterns for MIG inverse training.

All functions return:
    J : (3, Nx, Ny, Nz)  on the given device

Main entry:
    sample_current_pattern(Nx, Ny, Nz, device, X=None, Y=None, Z=None)

This keeps the SAME API your MIGInverseDataset expects.
"""


def _make_coords(Nx, Ny, Nz, device):
    xs = torch.linspace(-0.015, 0.015, Nx, device=device)   # 3 cm span
    ys = torch.linspace(-0.015, 0.015, Ny, device=device)
    zs = torch.linspace(0.0,    0.03,  Nz, device=device)   # 3 cm depth

    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    return X, Y, Z


# ---------- base building blocks ----------

def _gaussian_blob(X, Y, Z, cx, cy, cz, sx, sy, sz, amp=1.0):
    """
    General anisotropic 3D Gaussian.
    """
    dx = (X - cx) / sx
    dy = (Y - cy) / sy
    dz = (Z - cz) / sz
    r2 = dx * dx + dy * dy + dz * dz
    return amp * torch.exp(-0.5 * r2)


def _radial_unit_vector(X, Y, Z, cx, cy, cz, eps=1e-6):
    """
    Unit vector pointing radially outward from center.
    Returns (ux, uy, uz) each (Nx,Ny,Nz)
    """
    rx = X - cx
    ry = Y - cy
    rz = Z - cz
    r = torch.sqrt(rx * rx + ry * ry + rz * rz + eps)
    return rx / r, ry / r, rz / r


def _tangential_unit_vector_xy(X, Y, cx, cy, eps=1e-6):
    """
    Tangential (angular) vector in XY plane (like a swirl around z-axis).
    Returns (ux, uy) of shape (Nx,Ny,Nz).
    """
    rx = X - cx
    ry = Y - cy
    r = torch.sqrt(rx * rx + ry * ry + eps)
    # rotate (rx,ry) by +90 deg → (-ry, rx)
    tx = -ry / r
    ty = rx / r
    return tx, ty


# ---------- pattern types ----------

def make_healthy_like(Nx, Ny, Nz, device, X=None, Y=None, Z=None):
    """
    Healthy-ish: single smooth dipole-ish current in mid-myocardium; 
    oriented mostly along x with small y,z components.
    """
    if X is None or Y is None or Z is None:
        X, Y, Z = _make_coords(Nx, Ny, Nz, device)

    cx = torch.empty(1, device=device).uniform_(-0.005, 0.005).item()
    cy = torch.empty(1, device=device).uniform_(-0.005, 0.005).item()
    cz = torch.empty(1, device=device).uniform_(0.01, 0.02).item()

    sx = torch.empty(1, device=device).uniform_(0.003, 0.007).item()
    sy = torch.empty(1, device=device).uniform_(0.003, 0.007).item()
    sz = torch.empty(1, device=device).uniform_(0.003, 0.008).item()

    blob = _gaussian_blob(X, Y, Z, cx, cy, cz, sx, sy, sz, amp=1.0)

    # mostly x-directed, with slight y,z
    theta_y = torch.empty(1, device=device).uniform_(-0.2, 0.2).item()
    theta_z = torch.empty(1, device=device).uniform_(-0.2, 0.2).item()

    Jx = blob * math.cos(theta_y) * math.cos(theta_z)
    Jy = blob * math.sin(theta_y)
    Jz = blob * math.sin(theta_z)

    J = torch.stack([Jx, Jy, Jz], dim=0)  # (3,Nx,Ny,Nz)
    return J


def make_ischemia_like(Nx, Ny, Nz, device, X=None, Y=None, Z=None):
    """
    Ischemia-ish: start from healthy and suppress a localized patch -> “weak” region.
    """
    if X is None or Y is None or Z is None:
        X, Y, Z = _make_coords(Nx, Ny, Nz, device)

    J = make_healthy_like(Nx, Ny, Nz, device, X, Y, Z)

    # choose ischemic region
    cx = torch.empty(1, device=device).uniform_(-0.01, 0.01).item()
    cy = torch.empty(1, device=device).uniform_(-0.01, 0.01).item()
    cz = torch.empty(1, device=device).uniform_(0.005, 0.025).item()

    sx = torch.empty(1, device=device).uniform_(0.003, 0.008).item()
    sy = torch.empty(1, device=device).uniform_(0.003, 0.008).item()
    sz = torch.empty(1, device=device).uniform_(0.003, 0.010).item()

    lesion = _gaussian_blob(X, Y, Z, cx, cy, cz, sx, sy, sz, amp=1.0)

    # 0.1–0.4 factor in lesion region → reduced magnitude
    factor = 0.1 + 0.3 * lesion
    factor = 1.0 - 0.7 * lesion  # 1 where no lesion, ~0.3 where strong lesion
    J = J * factor.unsqueeze(0)  # (3,Nx,Ny,Nz)

    return J


def make_arrhythmia_like(Nx, Ny, Nz, device, X=None, Y=None, Z=None):
    """
    Arrhythmia-ish: swirling, multi-focal vortical pattern; high spatial complexity.
    """
    if X is None or Y is None or Z is None:
        X, Y, Z = _make_coords(Nx, Ny, Nz, device)

    # swirl center near heart center
    cx = torch.empty(1, device=device).uniform_(-0.003, 0.003).item()
    cy = torch.empty(1, device=device).uniform_(-0.003, 0.003).item()
    cz = torch.empty(1, device=device).uniform_(0.01, 0.02).item()

    # swirl radius ~ 1–1.5 cm
    base = _gaussian_blob(X, Y, Z, cx, cy, cz,
                          sx=0.010, sy=0.010, sz=0.008, amp=1.0)

    tx, ty = _tangential_unit_vector_xy(X, Y, cx, cy)
    # vertical modulation: sign changes along z
    phase_z = torch.empty(1, device=device).uniform_(0.0, 2 * math.pi).item()
    mod_z = torch.sin(5 * (Z - cz) / 0.03 + phase_z)

    Jx = base * tx * mod_z
    Jy = base * ty * mod_z
    Jz = 0.3 * base * mod_z  # small z component

    # add a second focal swirl occasionally
    if torch.rand(1, device=device).item() < 0.5:
        cx2 = cx + torch.empty(1, device=device).uniform_(-0.005, 0.005).item()
        cy2 = cy + torch.empty(1, device=device).uniform_(-0.005, 0.005).item()
        cz2 = cz + torch.empty(1, device=device).uniform_(-0.005, 0.005).item()
        base2 = _gaussian_blob(X, Y, Z, cx2, cy2, cz2,
                               sx=0.008, sy=0.008, sz=0.006, amp=0.7)
        tx2, ty2 = _tangential_unit_vector_xy(X, Y, cx2, cy2)
        mod_z2 = torch.cos(4 * (Z - cz2) / 0.03 + phase_z)
        Jx = Jx + base2 * tx2 * mod_z2
        Jy = Jy + base2 * ty2 * mod_z2
        Jz = Jz + 0.3 * base2 * mod_z2

    J = torch.stack([Jx, Jy, Jz], dim=0)
    return J


def make_conduction_block_like(Nx, Ny, Nz, device, X=None, Y=None, Z=None):
    """
    Conduction block-ish: strong current in one half of the heart, almost zero in the other.
    """
    if X is None or Y is None or Z is None:
        X, Y, Z = _make_coords(Nx, Ny, Nz, device)

    J = make_healthy_like(Nx, Ny, Nz, device, X, Y, Z)

    # random block plane normal along x or y
    if torch.rand(1, device=device).item() < 0.5:
        # x-split
        threshold = torch.empty(1, device=device).uniform_(-0.002, 0.002).item()
        mask = (X > threshold).float()
    else:
        # y-split
        threshold = torch.empty(1, device=device).uniform_(-0.002, 0.002).item()
        mask = (Y > threshold).float()

    # one side scaled down
    block_factor = 0.2 + 0.1 * torch.rand(1, device=device).item()  # 0.2–0.3
    factor = (1.0 - mask) + block_factor * mask  # 1 on one side, ~0.2 on the other
    J = J * factor.unsqueeze(0)
    return J


def make_noise_enhanced(Nx, Ny, Nz, device, X=None, Y=None, Z=None):
    """
    Purely synthetic random but *smooth-ish* field to regularize:
    combination of low-frequency sines and a weak gaussian.
    """
    if X is None or Y is None or Z is None:
        X, Y, Z = _make_coords(Nx, Ny, Nz, device)

    # scale xyz to nice ranges
    kx = torch.randint(1, 4, (1,), device=device).item()
    ky = torch.randint(1, 4, (1,), device=device).item()
    kz = torch.randint(1, 4, (1,), device=device).item()

    phase = 2 * math.pi * torch.rand(3, device=device)

    Jx = torch.sin(kx * math.pi * X / 0.03 + phase[0]) * \
         torch.cos(ky * math.pi * Y / 0.03 + phase[1])
    Jy = torch.sin(ky * math.pi * Y / 0.03 + phase[1]) * \
         torch.cos(kz * math.pi * Z / 0.03 + phase[2])
    Jz = torch.sin(kz * math.pi * Z / 0.03 + phase[2]) * \
         torch.cos(kx * math.pi * X / 0.03 + phase[0])

    # add a small gaussian envelope
    blob = _gaussian_blob(
        X, Y, Z,
        cx=0.0, cy=0.0, cz=0.015,
        sx=0.012, sy=0.012, sz=0.010,
        amp=1.0
    )

    Jx = 0.3 * Jx * blob
    Jy = 0.3 * Jy * blob
    Jz = 0.3 * Jz * blob

    J = torch.stack([Jx, Jy, Jz], dim=0)
    return J


# ---------- main sampler used by dataset ----------

def sample_current_pattern(Nx, Ny, Nz, device, X=None, Y=None, Z=None):
    """
    Unified entry point used by MIGInverseDataset.

    Randomly mixes between:
      - healthy-like
      - ischemia-like
      - arrhythmia-like
      - conduction-block-like
      - smooth noise patterns

    This makes the inverse problem richer / more realistic 
    for ISEF-level narrative.
    """

    if X is None or Y is None or Z is None:
        X, Y, Z = _make_coords(Nx, Ny, Nz, device)

    # pick pattern type
    r = torch.rand(1, device=device).item()

    if r < 0.25:
        J = make_healthy_like(Nx, Ny, Nz, device, X, Y, Z)
        pattern_type = "healthy_like"
    elif r < 0.5:
        J = make_ischemia_like(Nx, Ny, Nz, device, X, Y, Z)
        pattern_type = "ischemia_like"
    elif r < 0.75:
        J = make_arrhythmia_like(Nx, Ny, Nz, device, X, Y, Z)
        pattern_type = "arrhythmia_like"
    elif r < 0.9:
        J = make_conduction_block_like(Nx, Ny, Nz, device, X, Y, Z)
        pattern_type = "block_like"
    else:
        J = make_noise_enhanced(Nx, Ny, Nz, device, X, Y, Z)
        pattern_type = "noise_like"

    # small global scaling so magnitudes vary
    scale = 0.5 + torch.rand(1, device=device).item()  # 0.5–1.5
    J = scale * J

    # we don't return pattern_type to keep dataset API unchanged,
    # but you *could* store it in the future for labeled disease training.

    return J
