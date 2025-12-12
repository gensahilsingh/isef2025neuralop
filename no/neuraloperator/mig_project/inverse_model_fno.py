# inverse_model_fno.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models.fno import FNO  # neuraloperator's FNO


class MIGInverseFNO(nn.Module):
    """
    Inverse model: B sensors (3, Hs, Ws) → J volume (3, Nx, Ny, Nz)
    using official NeuralOperator FNO.
    """

    def __init__(
        self,
        Nx=32,
        Ny=32,
        Nz=16,
        Hs=16,
        Ws=16,
        latent_channels=24,
        fno_width=48,
        modes_x=12,
        modes_y=12,
        modes_z=8,
        fno_layers=4,
    ):
        super().__init__()

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Hs, self.Ws = Hs, Ws

        # 2D encoder over sensor plane B (3 channels: Bx, By, Bz)
        self.encoder2d = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # lift 2D latent → 3D latent & match FNO width
        self.lift3d = nn.Conv3d(latent_channels, fno_width, kernel_size=1)

        # FNO over 3D volume with coords
        self.fno = FNO(
            n_modes=(modes_x, modes_y, modes_z),
            in_channels=fno_width + 3,  # features + (x,y,z)
            out_channels=3,             # Jx, Jy, Jz
            hidden_channels=fno_width,
            n_layers=fno_layers,
        )

    def forward(self, B_plane, X, Y, Z):
        """
        B_plane: (B, 3, Hs, Ws)   – sensor plane field
        X,Y,Z  : (1, 1, Nx, Ny, Nz)  coords (broadcasted later)
        returns:
            J_pred: (B, 3, Nx, Ny, Nz)
        """
        Bsz = B_plane.size(0)

        # encode sensor plane
        latent2d = self.encoder2d(B_plane)  # (B, C_lat, Hs, Ws)

        # upsample to target spatial resolution in x,y
        latent2d_up = F.interpolate(
            latent2d,
            size=(self.Nx, self.Ny),
            mode="bilinear",
            align_corners=False,
        )  # (B, C_lat, Nx, Ny)

        # lift to 3D by repeating along z and passing through 1x1x1
        latent3d = latent2d_up.unsqueeze(-1).repeat(1, 1, 1, 1, self.Nz)
        latent3d = self.lift3d(latent3d)  # (B, fno_width, Nx, Ny, Nz)

        # broadcast coordinate grids to batch
        # X,Y,Z expected to be (1,1,Nx,Ny,Nz)
        assert X.dim() == 5 and X.shape[2:] == (self.Nx, self.Ny, self.Nz), \
            f"X must be (1,1,{self.Nx},{self.Ny},{self.Nz}), got {X.shape}"

        Xb = X.expand(Bsz, -1, -1, -1, -1)
        Yb = Y.expand(Bsz, -1, -1, -1, -1)
        Zb = Z.expand(Bsz, -1, -1, -1, -1)

        # concatenate features + coordinates
        fno_in = torch.cat([latent3d, Xb, Yb, Zb], dim=1)  # (B, fno_width+3, Nx,Ny,Nz)

        # run FNO
        J_pred = self.fno(fno_in)  # (B, 3, Nx,Ny,Nz)

        return J_pred
