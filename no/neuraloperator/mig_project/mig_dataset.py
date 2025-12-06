# mig_dataset.py

import torch
from torch.utils.data import Dataset

from mig_geometry import make_volume_coords, make_sensor_plane
from mig_patterns import sample_current_pattern
from mig_forward import forward_biot_savart_sensors


class MIGInverseDataset(Dataset):
    """
    Dataset for training the inverse model:
      Input:  B_sensors on 2D sensor grid (3,Hs,Ws)
      Output: J_true on 3D voxel grid (3,Nx,Ny,Nz)
      Label: disease class (0..3)
    Everything is generated on-the-fly from patterns + Biot-Savart.
    """
    def __init__(
        self,
        n_samples,
        Nx=32, Ny=32, Nz=16,
        dx=1e-3, dy=1e-3, dz=1e-3,
        Nsx=16, Nsy=16,
        sensor_offset=0.01,
        noise_sigma=0.05,
        device="cpu"
    ):
        super().__init__()
        self.n_samples = n_samples
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.Nsx, self.Nsy = Nsx, Nsy
        self.sensor_offset = sensor_offset
        self.noise_sigma = noise_sigma
        self.device = device

        # precompute volume coords
        self.X, self.Y, self.Z = make_volume_coords(
            Nx, Ny, Nz, dx, dy, dz, device=device
        )

        # precompute sensor positions
        self.sensor_coords, _, _ = make_sensor_plane(
            Nsx, Nsy, self.X, self.Y, self.Z,
            sensor_offset=sensor_offset,
            device=device
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 1. sample J + label
        J_true, label = sample_current_pattern(self.X, self.Y, self.Z, device=self.device)
        # 2. forward Biot-Savart: J -> B at sensors
        B_sensors = forward_biot_savart_sensors(
            J_true, self.X, self.Y, self.Z,
            self.sensor_coords,
            self.dx, self.dy, self.dz
        )   # (Ns,3)

        # 3. reshape sensors into 2D grid (Hs,Ws)
        Ns = self.Nsx * self.Nsy
        assert B_sensors.shape[0] == Ns
        B_grid = B_sensors.view(self.Nsx, self.Nsy, 3).permute(2, 0, 1)  # (3,Hs,Ws)

        # 4. normalize and add noise
        max_abs = B_grid.abs().max() + 1e-8
        B_norm = B_grid / max_abs
        noise = self.noise_sigma * torch.randn_like(B_norm)
        B_noisy = B_norm + noise

        sample = {
            "B": B_noisy,        # (3,Hs,Ws)
            "J": J_true,         # (3,Nx,Ny,Nz)
            "label": torch.tensor(label, dtype=torch.long)
        }
        return sample


class MIGClassifierDataset(Dataset):
    """
    Dataset for classifier training (forward model):
      Input: J_true (or J_pred) voxel fields
      Label: disease class
    For now, we use ground truth J; later you can swap to J_pred from inverse model.
    """
    def __init__(
        self,
        n_samples,
        Nx=32, Ny=32, Nz=16,
        dx=1e-3, dy=1e-3, dz=1e-3,
        device="cpu"
    ):
        super().__init__()
        self.n_samples = n_samples
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.device = device

        self.X, self.Y, self.Z = make_volume_coords(
            Nx, Ny, Nz, dx, dy, dz, device=device
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        J_true, label = sample_current_pattern(self.X, self.Y, self.Z, device=self.device)
        return {
            "J": J_true,
            "label": torch.tensor(label, dtype=torch.long)
        }
