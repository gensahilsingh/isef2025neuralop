# train_inverse_fno.py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from mig_dataset import MIGInverseDataset
from inverse_model_fno import MIGInverseFNO
from mig_forward import make_sensor_grid
from losses import inverse_physics_loss


def make_coords_3d(Nx, Ny, Nz, device):
    """
    Build coordinate grids X,Y,Z of shape (1,1,Nx,Ny,Nz)
    in meters.
    """
    xs = torch.linspace(-0.015, 0.015, Nx, device=device)
    ys = torch.linspace(-0.015, 0.015, Ny, device=device)
    zs = torch.linspace(0.0,   0.03,  Nz, device=device)

    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    X = X.unsqueeze(0).unsqueeze(0)  # (1,1,Nx,Ny,Nz)
    Y = Y.unsqueeze(0).unsqueeze(0)
    Z = Z.unsqueeze(0).unsqueeze(0)
    return X, Y, Z


def train_inverse_fno():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>> training inverse FNO on device:", device)

    # geometry / hyperparams
    Nx, Ny, Nz = 32, 32, 16
    Hs, Ws = 16, 16   # sensor plane
    Nsx, Nsy = Hs, Ws

    n_train = 3000
    n_val = 500
    batch_size = 4
    noise_sigma = 0.05

    # sensor coords (S,3) and 3D coords
    sensor_coords = make_sensor_grid(Nsx=Nxsx if (Nxsx:=Nsx) else 16, Nsy=Nsy, device=device)
    # lol python won't like that one-liner, let's be sane:
    sensor_coords = make_sensor_grid(Nsx=Nsx, Nsy=Nsy, device=device)

    X, Y, Z = make_coords_3d(Nx, Ny, Nz, device)

    print(f"volume: {Nx}×{Ny}×{Nz}, sensors: {Nsx}×{Nsy}={Nsx*Nsy}")
    print(f"train={n_train}, val={n_val}")
    print("starting training...")

    # datasets
    train_set = MIGInverseDataset(
        n_samples=n_train,
        Nx=Nx, Ny=Ny, Nz=Nz,
        Nsx=Nsx, Nsy=Nsy,
        noise_sigma=noise_sigma,
        device=device,
    )
    val_set = MIGInverseDataset(
        n_samples=n_val,
        Nx=Nx, Ny=Ny, Nz=Nz,
        Nsx=Nsx, Nsy=Nsy,
        noise_sigma=noise_sigma,
        device=device,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # model
    model = MIGInverseFNO(
        Nx=Nx, Ny=Ny, Nz=Nz,
        Hs=Hs, Ws=Ws,
        latent_channels=24,
        fno_width=48,
        modes_x=12, modes_y=12, modes_z=8,
        fno_layers=4,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    n_epochs = 50

    for epoch in range(1, n_epochs + 1):
        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0
        n_train_samples = 0

        for batch in train_loader:
            B_plane = batch["B"].to(device)  # (B,3,Hs,Ws)
            J_true = batch["J"].to(device)   # (B,3,Nx,Ny,Nz)

            optimizer.zero_grad()

            J_pred = model(B_plane, X, Y, Z)

            loss, logs = inverse_physics_loss(
                J_pred,
                J_true,
                sensor_coords,
                dx, dy, dz,
                lambda_J=1.0,
                lambda_B=1.0,
                lambda_div=0.01,
                lambda_smooth=0.001,
            )


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * B_plane.size(0)
            n_train_samples += B_plane.size(0)

        train_loss = train_loss_sum / n_train_samples

        # ---- VAL ----
        model.eval()
        val_loss_sum = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                B_plane = batch["B"].to(device)
                J_true = batch["J"].to(device)

                J_pred = model(B_plane, X, Y, Z)

                loss, _ = inverse_physics_loss(
                    J_pred=J_pred,
                    J_true=J_true,
                    B_true=B_plane,
                    sensor_coords=sensor_coords,
                    w_recon=1.0,
                    w_B=0.1
                )

                val_loss_sum += loss.item() * B_plane.size(0)
                n_val_samples += B_plane.size(0)

        val_loss = val_loss_sum / n_val_samples

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

    torch.save(model.state_dict(), "mig_inverse_fno.pt")
    print("saved inverse FNO model → mig_inverse_fno.pt")


if __name__ == "__main__":
    train_inverse_fno()
