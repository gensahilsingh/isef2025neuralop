# losses.py
import torch
import torch.nn as nn

from mig_forward import forward_biot_savart_sensors


def _finite_diff_central(field, dx, dim):
    """
    central difference along given dim.
    field: (B, C, Nx,Ny,Nz)
    returns: derivative on interior (B, C, Nx-2,Ny-2,Nz-2)
    """
    # we'll always slice interior 1:-1 in *all* dims later,
    # so just do derivative along `dim` but keep others full.
    if dim == 2:  # x
        return (field[:, :, 2:, 1:-1, 1:-1] - field[:, :, :-2, 1:-1, 1:-1]) / (2.0 * dx)
    elif dim == 3:  # y
        return (field[:, :, 1:-1, 2:, 1:-1] - field[:, :, 1:-1, :-2, 1:-1]) / (2.0 * dx)
    elif dim == 4:  # z
        return (field[:, :, 1:-1, 1:-1, 2:] - field[:, :, 1:-1, 1:-1, :-2]) / (2.0 * dx)
    else:
        raise ValueError("dim must be 2,3,4 for (B,C,Nx,Ny,Nz)")


def _divergence_penalty(J, dx, dy, dz):
    """
    J: (B,3,Nx,Ny,Nz)
    Returns scalar divergence loss using central differences in interior.
    """
    B, C, Nx, Ny, Nz = J.shape
    assert C == 3

    # interior region (for nice central diffs)
    # we will compute dJx/dx on (Nx-2,Ny-2,Nz-2), etc, then sum.
    Jx = J[:, 0:1, :, :, :]
    Jy = J[:, 1:2, :, :, :]
    Jz = J[:, 2:3, :, :, :]

    dJx_dx = _finite_diff_central(Jx, dx, dim=2)  # (B,1,Nx-2,Ny-2,Nz-2)
    dJy_dy = _finite_diff_central(Jy, dy, dim=3)  # (B,1,Nx-2,Ny-2,Nz-2)
    dJz_dz = _finite_diff_central(Jz, dz, dim=4)  # (B,1,Nx-2,Ny-2,Nz-2)

    div = dJx_dx + dJy_dy + dJz_dz  # (B,1,Nx-2,Ny-2,Nz-2)
    return torch.mean(div ** 2)


def _smoothness_penalty(J):
    """
    simple quadratic smoothness: ||∇J||^2 over all spatial directions.
    J: (B,3,Nx,Ny,Nz)
    """
    B, C, Nx, Ny, Nz = J.shape

    # forward finite differences with zero at last slice
    diff_x = J[:, :, 1:, :, :] - J[:, :, :-1, :, :]
    diff_y = J[:, :, :, 1:, :] - J[:, :, :, :-1, :]
    diff_z = J[:, :, :, :, 1:] - J[:, :, :, :, :-1]

    sx = torch.mean(diff_x ** 2)
    sy = torch.mean(diff_y ** 2)
    sz = torch.mean(diff_z ** 2)

    return sx + sy + sz


def inverse_physics_loss(
    J_pred,
    J_true,
    B_true_grid,
    sensor_coords,
    dx,
    dy,
    dz,
    **kwargs,
):
    """
    Master loss for inverse MIG problem.

    Inputs:
      J_pred      : (B,3,Nx,Ny,Nz) reconstructed by FNO
      J_true      : (B,3,Nx,Ny,Nz) ground truth phantom
      B_true_grid : (B,3,Hs,Ws)    (from dataset, not used directly here)
      sensor_coords: (S,3)         sensor positions
      dx,dy,dz    : voxel spacings (for div / smoothness weighting)

    kwargs (all optional, we support multiple names to not break old code):
      lambda_J / lambda_j / lambda_rec
      lambda_B / lambda_b / lambda_phys
      lambda_div
      lambda_smooth / lambda_s

    Returns:
      loss (scalar), logs (dict of individual terms)
    """

    # ---- pull weights from kwargs with multiple fallbacks
    lam_J = kwargs.get("lambda_J", kwargs.get("lambda_j", kwargs.get("lambda_rec", 1.0)))
    lam_B = kwargs.get("lambda_B", kwargs.get("lambda_b", kwargs.get("lambda_phys", 1.0)))
    lam_div = kwargs.get("lambda_div", 0.01)
    lam_smooth = kwargs.get("lambda_smooth", kwargs.get("lambda_s", 0.001))

    # --- 1) direct J reconstruction ---
    J_loss = nn.functional.mse_loss(J_pred, J_true)

    # --- 2) Biot–Savart consistency ---
    # B_pred_sensors, B_true_sensors: (B,S,3)
    B_pred_sensors = forward_biot_savart_sensors(J_pred, sensor_coords)
    B_true_sensors = forward_biot_savart_sensors(J_true, sensor_coords)
    B_loss = nn.functional.mse_loss(B_pred_sensors, B_true_sensors)

    # --- 3) divergence penalty ---
    div_loss = _divergence_penalty(J_pred, dx, dy, dz)

    # --- 4) smoothness penalty ---
    smooth_loss = _smoothness_penalty(J_pred)

    # total
    loss = lam_J * J_loss + lam_B * B_loss + lam_div * div_loss + lam_smooth * smooth_loss

    logs = {
        "J_loss": J_loss.detach().item(),
        "B_loss": B_loss.detach().item(),
        "div_loss": div_loss.detach().item(),
        "smooth_loss": smooth_loss.detach().item(),
        "lam_J": lam_J,
        "lam_B": lam_B,
        "lam_div": lam_div,
        "lam_smooth": lam_smooth,
    }

    return loss, logs
