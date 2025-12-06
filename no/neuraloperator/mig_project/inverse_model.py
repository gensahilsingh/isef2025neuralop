# inverse_model.py

import torch
import torch.nn as nn

from fno3d import FNO3d


class SensorEncoder2D(nn.Module):
    """
    Simple 2D CNN encoder for sensor-plane B-field.
    Input: (batch,3,Hs,Ws)
    Output: (batch, C_latent, Hs,Ws)
    """
    def __init__(self, in_channels=3, latent_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, latent_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class MIGInverseModel(nn.Module):
    """
    Inverse model:
      Input: B_sensors on plane (3,Hs,Ws)
      Output: J_pred voxel currents (3,Nx,Ny,Nz)
    Architecture:
      1) 2D encoder on B-plane -> latent (C_lat,Hs,Ws)
      2) broadcast / interpolate latent to (C_lat,Nx,Ny,Nz)
      3) concat coordinate channels
      4) 3D FNO on (C_lat+3) -> 3-channel J
    """

    def __init__(
        self,
        Nx,
        Ny,
        Nz,
        Hs,
        Ws,
        latent_channels=16,
        fno_width=32,
        fno_layers=4,
        modes_x=12,
        modes_y=12,
        modes_z=8
    ):
        super().__init__()
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Hs, self.Ws = Hs, Ws

        self.encoder2d = SensorEncoder2D(in_channels=3, latent_channels=latent_channels)

        # FNO expects input channels = latent + coord (x,y,z)
        in_channels_fno = latent_channels + 3
        self.fno3d = FNO3d(
            in_channels=in_channels_fno,
            out_channels=3,
            modes_x=modes_x,
            modes_y=modes_y,
            modes_z=modes_z,
            width=fno_width,
            n_layers=fno_layers
        )

    def lift_to_volume(self, latent_2d):
        """
        latent_2d: (batch,C_lat,Hs,Ws)
        Returns latent_3d: (batch,C_lat,Nx,Ny,Nz) by interpolation + broadcast in z.
        """
        batch, C, Hs, Ws = latent_2d.shape
        # interpolate spatially to (Nx,Ny)
        latent_xy = torch.nn.functional.interpolate(
            latent_2d,
            size=(self.Nx, self.Ny),
            mode="bilinear",
            align_corners=False
        )  # (b,C,Nx,Ny)

        latent_xy = latent_xy.unsqueeze(-1).repeat(1, 1, 1, 1, self.Nz)  # (b,C,Nx,Ny,Nz)
        return latent_xy

    def make_coord_channels(self, X, Y, Z):
        """
        X,Y,Z: (Nx,Ny,Nz) in meters
        Returns coord_channels: (1,3,Nx,Ny,Nz) normalized.
        """
        # normalize coords to [-1,1]
        Xn = 2 * (X - X.min()) / (X.max() - X.min() + 1e-8) - 1
        Yn = 2 * (Y - Y.min()) / (Y.max() - Y.min() + 1e-8) - 1
        Zn = 2 * (Z - Z.min()) / (Z.max() - Z.min() + 1e-8) - 1

        coords = torch.stack([Xn, Yn, Zn], dim=0).unsqueeze(0)  # (1,3,Nx,Ny,Nz)
        return coords

    def forward(self, B_plane, X, Y, Z):
        """
        B_plane: (batch,3,Hs,Ws)
        X,Y,Z:   (Nx,Ny,Nz) coordinate tensors (on same device)
        """
        batch = B_plane.shape[0]
        device = B_plane.device
        X = X.to(device)
        Y = Y.to(device)
        Z = Z.to(device)

        # 1) encode sensor plane
        latent_2d = self.encoder2d(B_plane)  # (b,C_lat,Hs,Ws)

        # 2) lift to volume
        latent_3d = self.lift_to_volume(latent_2d)  # (b,C_lat,Nx,Ny,Nz)

        # 3) add coord channels
        coord_channels = self.make_coord_channels(X, Y, Z).to(device)  # (1,3,Nx,Ny,Nz)
        coord_channels = coord_channels.repeat(batch, 1, 1, 1, 1)     # (b,3,Nx,Ny,Nz)

        fno_input = torch.cat([latent_3d, coord_channels], dim=1)      # (b,C_lat+3,Nx,Ny,Nz)

        # 4) FNO
        J_pred = self.fno3d(fno_input)  # (b,3,Nx,Ny,Nz)
        return J_pred
