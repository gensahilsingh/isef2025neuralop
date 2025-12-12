# train_inverse_v2.py
#
# trains inverse model on realistic MIG disease patterns
# includes:
# - disease J patterns (healthy, ischemia, arrhythmia)
# - forward B-field simulation
# - normalization of J and B
# - divergence physics loss
# - improved architecture (latent=24, width=48)

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from inverse_model import MIGInverseModel
from mig_geometry import make_volume_coords, make_sensor_plane
from mig_forward import compute_B_field_from_J
from disease_patterns import generate_disease_sample


# ---------------------------------------------------------
# Dataset v2 — disease currents + forward field + normalize
# ---------------------------------------------------------
class MIGInverseDataset_v2(torch.utils.data.Dataset):
    def __init__(self, n_samples, Nx, Ny, Nz,
                 dx, dy, dz, Nsx, Nsy, sensor_offset, device):
        
        self.n_samples = n_samples
        self.device = device

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Nsx, self.Nsy = Nsx, Nsy

        # coordinates and sensor layout
        X, Y, Z = make_volume_coords(Nx, Ny, Nz, dx, dy, dz, device)
        sensor_coords, _, _ = make_sensor_plane(
            Nsx, Nsy, X, Y, Z, sensor_offset=sensor_offset, device=device
        )

        self.X, self.Y, self.Z = X, Y, Z
        self.sensor_coords = sensor_coords

        # normalization constants
        self.J_scale = 50.0
        self.B_scale = 50.0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # disease current map (3, Nx, Ny, Nz)
        J = generate_disease_sample(self.Nx, self.Ny, self.Nz, self.device)[0]

        # forward magnetic field → (3, Hs, Ws)
        B = compute_B_field_from_J(J, self.X, self.Y, self.Z, self.sensor_coords)

        # normalize both
        Jn = J / self.J_scale
        Bn = B / self.B_scale

        return {
            "J": Jn,
            "B": Bn
        }


# ---------------------------------------------------------
# Divergence loss — FIXED version with correct padding
# ---------------------------------------------------------
def divergence_loss(J_pred):
    """
    J_pred shape: (B,3,Nx,Ny,Nz)
    Computes ∂Jx/∂x + ∂Jy/∂y + ∂Jz/∂z
    using centered finite differencing.
    """

    Jx = J_pred[:, 0]
    Jy = J_pred[:, 1]
    Jz = J_pred[:, 2]

    # centered difference
    dJx_dx = Jx[:, 2:, :, :] - Jx[:, :-2, :, :]
    dJy_dy = Jy[:, :, 2:, :] - Jy[:, :, :-2, :]
    dJz_dz = Jz[:, :, :, 2:] - Jz[:, :, :, :-2]

    # pad edges to maintain original shape
    dJx_dx = torch.nn.functional.pad(dJx_dx, (0,0, 0,0, 1,1))
    dJy_dy = torch.nn.functional.pad(dJy_dy, (0,0, 1,1, 0,0))
    dJz_dz = torch.nn.functional.pad(dJz_dz, (1,1, 0,0, 0,0))

    div = dJx_dx + dJy_dy + dJz_dz
    return torch.mean(torch.abs(div))


# ---------------------------------------------------------
# Combined reconstruction + physics loss
# ---------------------------------------------------------
def loss_fn(J_pred, J_true):
    mse = torch.mean((J_pred - J_true)**2)
    div = divergence_loss(J_pred)
    return mse + 0.01 * div


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_inverse_v2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>> train_inverse_v2 starting on device:", device)

    # geometry
    Nx, Ny, Nz = 32, 32, 16
    dx = dy = dz = 1e-3
    Nsx, Nsy = 16, 16
    sensor_offset = 0.01

    # dataset size
    n_train = 6000
    n_val = 1000
    batch_size = 4

    # load coords once
    X, Y, Z = make_volume_coords(Nx, Ny, Nz, dx, dy, dz, device)
    sensor_coords, _, _ = make_sensor_plane(
        Nsx, Nsy, X, Y, Z, sensor_offset=sensor_offset, device=device
    )

    # model — upgraded architecture
    model = MIGInverseModel(
        Nx=Nx, Ny=Ny, Nz=Nz,
        Hs=Nsx, Ws=Nsy,
        latent_channels=24,
        fno_width=48,
        fno_layers=4,
        modes_x=12, modes_y=12, modes_z=8
    ).to(device)

    # datasets
    train_set = MIGInverseDataset_v2(
        n_samples=n_train,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        Nsx=Nsx, Nsy=Nsy,
        sensor_offset=sensor_offset,
        device=device
    )

    val_set = MIGInverseDataset_v2(
        n_samples=n_val,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        Nsx=Nsx, Nsy=Nsy,
        sensor_offset=sensor_offset,
        device=device
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-4
    )

    print(">>> training inverse model v2...")

    EPOCHS = 100

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            J_true = batch["J"].to(device)
            B = batch["B"].to(device)

            optimizer.zero_grad()
            J_pred = model(B, X, Y, Z)
            loss = loss_fn(J_pred, J_true)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B.size(0)

        scheduler.step()
        train_loss /= len(train_set)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                J_true = batch["J"].to(device)
                B = batch["B"].to(device)
                J_pred = model(B, X, Y, Z)
                loss = loss_fn(J_pred, J_true)
                val_loss += loss.item() * B.size(0)

        val_loss /= len(val_set)

        print(f"epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    torch.save(model.state_dict(), "mig_inverse_fno_v2_diseases.pt")
    print("\n>>> saved: mig_inverse_fno_v2_diseases.pt")


if __name__ == "__main__":
    train_inverse_v2()
