# train_inverse.py (balanced / sweet spot version)

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from mig_dataset import MIGInverseDataset
from inverse_model import MIGInverseModel
from mig_geometry import make_volume_coords, make_sensor_plane
from losses import inverse_physics_loss


def train_inverse():
    print(">>> FILE LOADED")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    # geometry / hyperparams
    Nx, Ny, Nz = 32, 32, 16
    dx = dy = dz = 1e-3
    Nsx, Nsy = 16, 16
    sensor_offset = 0.01

    # GOOD SWEET SPOT: larger than original but not crazy heavy
    n_train = 3000
    n_val = 500

    # batch_size=4 may work on 4060 Ti; if OOM, set to 2
    batch_size = 4
    noise_sigma = 0.05

    # build coordinate grids (on device)
    X, Y, Z = make_volume_coords(Nx, Ny, Nz, dx, dy, dz, device=device)
    sensor_coords, Hs, Ws = make_sensor_plane(
        Nsx, Nsy, X, Y, Z, sensor_offset=sensor_offset, device=device
    )

    print(f"volume: {Nx}×{Ny}×{Nz}, sensors: {Nsx}×{Nsy}={Nsx*Nsy}")
    print(f"train={n_train}, val={n_val}")

    # dataset
    train_set = MIGInverseDataset(
        n_samples=n_train,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        Nsx=Nsx, Nsy=Nsy,
        sensor_offset=sensor_offset,
        noise_sigma=noise_sigma,
        device=device
    )
    val_set = MIGInverseDataset(
        n_samples=n_val,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        Nsx=Nsx, Nsy=Nsy,
        sensor_offset=sensor_offset,
        noise_sigma=noise_sigma,
        device=device
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # SWEET-SPOT MODEL SIZE (not too small, not too huge)
    model = MIGInverseModel(
        Nx=Nx, Ny=Ny, Nz=Nz,
        Hs=Nsx, Ws=Nsy,
        latent_channels=20,
        fno_width=36,
        fno_layers=4,
        modes_x=12, modes_y=12, modes_z=8
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # cosine schedule (good for smooth convergence)
    n_epochs = 150
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-4
    )

    print("starting training...")

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_samples = 0

        for batch in train_loader:
            B = batch["B"].to(device)
            J_true = batch["J"].to(device)

            optimizer.zero_grad()

            J_pred = model(B, X, Y, Z)

            loss, _ = inverse_physics_loss(
                J_pred, J_true, B,
                X, Y, Z, sensor_coords,
                dx, dy, dz,
                lambda_phys=0.12,  # slightly stronger than original
                lambda_div=0.01
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * B.size(0)
            n_train_samples += B.size(0)

        scheduler.step()
        train_loss = train_loss_sum / n_train_samples

        # validation
        model.eval()
        val_loss_sum = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                B = batch["B"].to(device)
                J_true = batch["J"].to(device)
                J_pred = model(B, X, Y, Z)

                loss, _ = inverse_physics_loss(
                    J_pred, J_true, B,
                    X, Y, Z, sensor_coords,
                    dx, dy, dz,
                    lambda_phys=0.12,
                    lambda_div=0.01
                )

                val_loss_sum += loss.item() * B.size(0)
                n_val_samples += B.size(0)

        val_loss = val_loss_sum / n_val_samples

        lr_current = scheduler.get_last_lr()[0]

        print(
            f"epoch {epoch:03d} | lr={lr_current:.5f} | "
            f"train={train_loss:.5f} | val={val_loss:.5f}"
        )

    torch.save(model.state_dict(), "mig_inverse_fno_v2.pt")
    print("saved model → mig_inverse_fno_v2.pt")


if __name__ == "__main__":
    train_inverse()
