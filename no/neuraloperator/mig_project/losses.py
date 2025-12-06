# losses.py

import torch
import torch.nn.functional as F

from mig_forward import forward_biot_savart_sensors


def divergence_3d(J, dx, dy, dz):
    """
    J: (batch,3,Nx,Ny,Nz)
    Returns div: (batch,Nx-2,Ny-2,Nz-2)
    """
    Jx = J[:, 0]
    Jy = J[:, 1]
    Jz = J[:, 2]

    dJx_dx = (Jx[:, 2:, 1:-1, 1:-1] - Jx[:, :-2, 1:-1, 1:-1]) / (2 * dx)
    dJy_dy = (Jy[:, 1:-1, 2:, 1:-1] - Jy[:, 1:-1, :-2, 1:-1]) / (2 * dy)
    dJz_dz = (Jz[:, 1:-1, 1:-1, 2:] - Jz[:, 1:-1, 1:-1, :-2]) / (2 * dz)

    div = dJx_dx + dJy_dy + dJz_dz
    return div


def inverse_physics_loss(
    J_pred,
    J_true,
    B_input,
    X,
    Y,
    Z,
    sensor_coords,
    dx,
    dy,
    dz,
    lambda_phys=0.1,
    lambda_div=0.01
):
    """
    Physics-informed loss for inverse model.

    J_pred: (b,3,Nx,Ny,Nz)
    J_true: (b,3,Nx,Ny,Nz)
    B_input: (b,3,Hs,Ws)  -> B at sensors, normalized
    X,Y,Z: (Nx,Ny,Nz)
    sensor_coords: (Ns,3)
    """
    device = J_pred.device
    batch = J_pred.shape[0]

    # L_data: MSE between J_pred and J_true
    L_data = F.mse_loss(J_pred, J_true)

    # L_phys: Biot-Savart consistency (per sample)
    L_phys_accum = 0.0
    for i in range(batch):
        Ji = J_pred[i]  # (3,Nx,Ny,Nz)
        Bi_pred_sensors = forward_biot_savart_sensors(
            Ji, X, Y, Z, sensor_coords, dx, dy, dz
        )  # (Ns,3)

        # reshape B_input[i] back to (Ns,3) for comparison
        b_i = B_input[i]  # (3,Hs,Ws)
        _, Hs, Ws = b_i.shape
        Ns = Hs * Ws
        b_flat = b_i.permute(1, 2, 0).reshape(Ns, 3)   # (Ns,3)

        # both are normalized-ish; MSE
        L_phys_accum += F.mse_loss(Bi_pred_sensors, b_flat)

    L_phys = L_phys_accum / batch

    # L_div: divergence penalty
    div = divergence_3d(J_pred, dx, dy, dz)
    L_div = torch.mean(div ** 2)

    total = L_data + lambda_phys * L_phys + lambda_div * L_div
    return total, {"L_data": L_data.item(), "L_phys": L_phys.item(), "L_div": L_div.item()}
