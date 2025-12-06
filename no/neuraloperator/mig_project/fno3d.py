# fno3d.py

import torch
import torch.nn as nn
import torch.fft


class SpectralConv3d(nn.Module):
    """
    3D Fourier spectral convolution layer.
    Keeps only a limited number of modes in each dimension.
    """

    def __init__(self, in_channels, out_channels, modes_x, modes_y, modes_z):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        # complex weights: real + imag parts
        self.weight = nn.Parameter(
            torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                modes_z,
                dtype=torch.cfloat
            )
            * 0.02
        )

    def compl_mul3d(self, input, weights):
        # (batch,in_c, X,Y,Z) x (in_c,out_c,mx,my,mz) -> (batch,out_c,X,Y,Z) in Fourier domain
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        """
        x: (batch, in_channels, Nx,Ny,Nz)
        """
        batchsize, _, Nx, Ny, Nz = x.shape

        # FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))  # (b,in_c,Nx,Ny,Nz//2+1)

        # allocate output in Fourier space
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            Nx,
            Ny,
            Nz // 2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )

        mx = min(self.modes_x, Nx)
        my = min(self.modes_y, Ny)
        mz = min(self.modes_z, Nz // 2 + 1)

        out_ft[:, :, :mx, :my, :mz] = self.compl_mul3d(
            x_ft[:, :, :mx, :my, :mz],
            self.weight[:, :, :mx, :my, :mz]
        )

        # inverse FFT
        x_out = torch.fft.irfftn(out_ft, s=(Nx, Ny, Nz), dim=(-3, -2, -1))
        return x_out


class FNO3d(nn.Module):
    """
    Basic 3D FNO network:
      Input: (batch, in_channels, Nx,Ny,Nz)
      Output: (batch, out_channels, Nx,Ny,Nz)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        modes_x=12,
        modes_y=12,
        modes_z=8,
        width=32,
        n_layers=4
    ):
        super().__init__()

        self.width = width
        self.n_layers = n_layers

        self.fc0 = nn.Linear(in_channels, width)

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                SpectralConv3d(width, width, modes_x, modes_y, modes_z)
            )
            self.ws.append(nn.Conv3d(width, width, kernel_size=1))

        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, out_channels)

        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: (batch, in_channels, Nx,Ny,Nz)
        """
        batchsize, in_channels, Nx, Ny, Nz = x.shape

        # move channel to feature dim, apply fc0
        x = x.permute(0, 2, 3, 4, 1)  # (b,Nx,Ny,Nz,in_c)
        x = self.fc0(x)               # (b,Nx,Ny,Nz,width)
        x = x.permute(0, 4, 1, 2, 3)  # (b,width,Nx,Ny,Nz)

        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = self.activation(x1 + x2)

        # project back to out_channels
        x = x.permute(0, 2, 3, 4, 1)  # (b,Nx,Ny,Nz,width)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)               # (b,Nx,Ny,Nz,out_c)
        x = x.permute(0, 4, 1, 2, 3)  # (b,out_c,Nx,Ny,Nz)
        return x
