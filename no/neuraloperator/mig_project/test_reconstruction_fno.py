# test_reconstruction_fno.py

import torch
from torch.utils.data import DataLoader

from mig_dataset import MIGInverseDataset
from inverse_model_fno import MIGInverseFNO
from mig_forward import forward_biot_savart_sensors


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    # geometry
    Nx, Ny, Nz = 32, 32, 16
    dx = dy = dz = 1e-3
    Nsx, Nsy = 16, 16

    # load sensor coords (same as in train_inverse_fno)
    from mig_forward import make_sensor_grid
    sensor_coords = make_sensor_grid(Nsx, Nsy, device=device)  # (256,3)

    # dataset with NEW patterns
    n_test = 100
    test_set = MIGInverseDataset(
        n_samples=n_test,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        Nsx=Nsx, Nsy=Nsy,
        noise_sigma=0.05,
        device=device,
    )
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

    # load inverse FNO
    model = MIGInverseFNO(
        Nx=Nx, Ny=Ny, Nz=Nz,
        Hs=Nsx, Ws=Nsy,
        latent_channels=24,
        fno_width=48,
        modes_x=12,
        modes_y=12,
        modes_z=8,
        fno_layers=4,
    ).to(device)

    ckpt_path = "mig_inverse_fno.pt"   # or whatever you used in train_inverse_fno
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"✓ loaded inverse FNO from {ckpt_path}")

    mse_list = []
    rel_list = []

    for batch_idx, batch in enumerate(test_loader, start=1):
        B = batch["B"].to(device)      # (B,3,Hs,Ws)
        J_true = batch["J"].to(device) # (B,3,Nx,Ny,Nz)

        # coordinates used by your geometry code
        X = test_set.X.to(device)
        Y = test_set.Y.to(device)
        Z = test_set.Z.to(device)

        J_pred = model(B, X, Y, Z)     # (B,3,Nx,Ny,Nz)

        # per-batch mse
        mse = torch.mean((J_pred - J_true) ** 2, dim=(1, 2, 3, 4))  # (B,)
        # relative error ||J_pred - J_true|| / ||J_true||
        num = torch.sqrt(torch.sum((J_pred - J_true) ** 2, dim=(1, 2, 3, 4)))
        den = torch.sqrt(torch.sum(J_true ** 2, dim=(1, 2, 3, 4))) + 1e-8
        rel = num / den

        mse_list.append(mse.cpu())
        rel_list.append(rel.cpu())

    mse_all = torch.cat(mse_list)
    rel_all = torch.cat(rel_list)

    print("========================================")
    print(f"Test samples     : {n_test}")
    print(f"Mean MSE(J)      : {mse_all.mean().item():.6e}")
    print(f"Median MSE(J)    : {mse_all.median().item():.6e}")
    print(f"Mean Rel Error   : {rel_all.mean().item():.4f}")
    print(f"Median Rel Error : {rel_all.median().item():.4f}")

    # also test Biot–Savart consistency on a few samples
    print("\nChecking Biot–Savart consistency on first batch...")
    batch = next(iter(test_loader))
    B_grid = batch["B"].to(device)
    J_true = batch["J"].to(device)

    X = test_set.X.to(device)
    Y = test_set.Y.to(device)
    Z = test_set.Z.to(device)

    J_pred = model(B_grid, X, Y, Z)

    B_true_s = forward_biot_savart_sensors(J_true, sensor_coords)  # (B,S,3)
    B_pred_s = forward_biot_savart_sensors(J_pred, sensor_coords)  # (B,S,3)

    B_mse = torch.mean((B_pred_s - B_true_s) ** 2).item()
    print(f"Biot–Savart sensor MSE: {B_mse:.6e}")


if __name__ == "__main__":
    main()
