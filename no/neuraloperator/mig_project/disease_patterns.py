import torch
import torch.nn.functional as F


def _smooth_field(shape, device, scale=1.0):
    Nx, Ny, Nz = shape
    x = torch.randn(1, 1, Nx, Ny, Nz, device=device)
    x = F.avg_pool3d(x, kernel_size=5, stride=1, padding=2)[0,0]
    x = x / (x.abs().max() + 1e-6) * scale
    return x


def make_healthy(Nx, Ny, Nz, device, strength=1.0):
    base = _smooth_field((Nx, Ny, Nz), device, scale=strength)

    gx = torch.linspace(-1, 1, Nx, device=device)[:, None, None]
    gy = torch.linspace(-1, 1, Ny, device=device)[None, :, None]
    gz = torch.linspace(-1, 1, Nz, device=device)[None, None, :]

    Jx = base * gx
    Jy = base * gy
    Jz = base * gz

    return torch.stack([Jx, Jy, Jz], dim=0)  # (3,Nx,Ny,Nz)


def make_ischemia(Nx, Ny, Nz, device):
    J = make_healthy(Nx, Ny, Nz, device)

    cx, cy, cz = Nx//2, Ny//2, Nz//2
    rx, ry, rz = Nx//5, Ny//5, Nz//5

    X = torch.arange(Nx, device=device)[:, None, None]
    Y = torch.arange(Ny, device=device)[None, :, None]
    Z = torch.arange(Nz, device=device)[None, None, :]

    mask = ((X - cx).abs() < rx) & ((Y - cy).abs() < ry) & ((Z - cz).abs() < rz)

    J[:, mask] *= 0.15
    return J


def make_arrhythmia(Nx, Ny, Nz, device):
    J = torch.randn(3, Nx, Ny, Nz, device=device)
    J = F.avg_pool3d(J.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]

    x = torch.linspace(0, 3.14, Nx, device=device)[:, None, None]
    y = torch.linspace(0, 3.14, Ny, device=device)[None, :, None]
    z = torch.linspace(0, 3.14, Nz, device=device)[None, None, :]

    wave = torch.sin(x) * torch.cos(y) * torch.sin(z)

    J = J + torch.stack([wave, -wave, wave * 0.5], dim=0)

    return J * 1.2


def make_block(Nx, Ny, Nz, device):
    """
    conduction block: HALF of the heart barely conducts.
    We need mask shape → (Nx,Ny,Nz)
    """
    J = make_healthy(Nx, Ny, Nz, device, strength=1.2)

    X = torch.arange(Nx, device=device)[:, None, None]
    # broadcast (Nx,1,1) → (Nx,Ny,Nz)
    Xb = X.expand(Nx, Ny, Nz)

    block_region = Xb > (Nx // 2)   # (Nx,Ny,Nz)

    J[:, block_region] *= 0.1
    return J


def make_scar(Nx, Ny, Nz, device):
    """
    scar: random patches suppressed.
    mask must be (Nx,Ny,Nz)
    """
    J = make_healthy(Nx, Ny, Nz, device)

    mask = (torch.rand(Nx, Ny, Nz, device=device) < 0.25)  # (Nx,Ny,Nz)
    J[:, mask] *= 0.05

    return J


def generate_disease_sample(Nx, Ny, Nz, device):
    r = torch.randint(0, 5, (1,), device=device).item()

    if r == 0:
        return make_healthy(Nx, Ny, Nz, device), 0
    elif r == 1:
        return make_ischemia(Nx, Ny, Nz, device), 1
    elif r == 2:
        return make_arrhythmia(Nx, Ny, Nz, device), 2
    elif r == 3:
        return make_block(Nx, Ny, Nz, device), 3
    else:
        return make_scar(Nx, Ny, Nz, device), 4
