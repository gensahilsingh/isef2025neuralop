import torch
from torch.utils.data import Dataset

from disease_patterns import generate_disease_sample
from mig_geometry import make_volume_coords, make_sensor_plane
from mig_forward import compute_B_field_from_J


class DiseaseDataset(Dataset):
    def __init__(self, n_samples, Nx, Ny, Nz,
                 dx, dy, dz,
                 Nsx, Nsy,
                 sensor_offset,
                 device):
        
        self.n_samples = n_samples
        self.device = device

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Nsx, self.Nsy = Nsx, Nsy

        # precompute coordinates + sensor layout
        X, Y, Z = make_volume_coords(Nx, Ny, Nz, dx, dy, dz, device=device)
        sensor_coords, _, _ = make_sensor_plane(
            Nsx, Nsy, X, Y, Z, sensor_offset=sensor_offset, device=device
        )

        self.X = X
        self.Y = Y
        self.Z = Z
        self.sensor_coords = sensor_coords

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # generate myocardial currents + label
        J_true, label = generate_disease_sample(
            self.Nx, self.Ny, self.Nz, self.device
        )  # J_true: (3,Nx,Ny,Nz)

        # compute synthetic MAGNETIC FIELD at sensors
        B = compute_B_field_from_J(
            J_true, self.X, self.Y, self.Z, self.sensor_coords
        )  # (3,Hs,Ws)

        return {
            "J": J_true,              # ground-truth voxel currents
            "B": B,                   # magnetic plane
            "label": torch.tensor(label, dtype=torch.long, device=self.device)
        }
