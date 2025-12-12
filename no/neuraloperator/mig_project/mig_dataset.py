# mig_dataset.py
import torch
from torch.utils.data import Dataset

from mig_forward import forward_biot_savart_sensors, make_sensor_grid
from mig_patterns import sample_current_pattern


class MIGInverseDataset(Dataset):
    """
    Simulation dataset for the inverse problem:
        J_true (3, Nx, Ny, Nz)  →  B_sensors (3, Nsx, Nsy)

    This is used by BOTH:
      - train_inverse.py
      - train_inverse_fno.py

    Shapes:
      J_true:      (3, Nx, Ny, Nz)
      B_grid:      (3, Nsx, Nsy)   # sensor plane, for the network input
      sensor_grid: (Nsx*Nsy, 3)    # (S,3) = (256,3)
      coords X,Y,Z: (1,1,Nx,Ny,Nz) # for models that need coordinates
    """

    def __init__(
        self,
        n_samples: int,
        Nx: int,
        Ny: int,
        Nz: int,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        Nsx: int = 16,
        Nsy: int = 16,
        sensor_offset: float = 0.06,
        noise_sigma: float = 0.05,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_samples = n_samples

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Nsx, self.Nsy = Nsx, Nsy
        self.dx, self.dy, self.dz = dx, dy, dz
        self.noise_sigma = noise_sigma
        self.device = torch.device(device)

        # --------------------------------------------------
        # volume coordinates (match what you're using elsewhere)
        # --------------------------------------------------
        xs = torch.linspace(-0.015, 0.015, Nx, device=self.device)
        ys = torch.linspace(-0.015, 0.015, Ny, device=self.device)
        zs = torch.linspace(0.0,     0.03, Nz, device=self.device)

        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
        # store as (1,1,Nx,Ny,Nz) for convenience in models
        self.X = X.unsqueeze(0).unsqueeze(0)  # (1,1,Nx,Ny,Nz)
        self.Y = Y.unsqueeze(0).unsqueeze(0)
        self.Z = Z.unsqueeze(0).unsqueeze(0)

        # --------------------------------------------------
        # sensor grid (Nsx*Nsy, 3) at plane z = sensor_offset
        # we ignore sensor_offset argument and use your make_sensor_grid
        # which already uses z=0.06; that's fine as long as it's consistent.
        # --------------------------------------------------
        self.sensor_coords = make_sensor_grid(Nsx, Nsy, device=self.device)  # (S,3)
        self.S = self.sensor_coords.shape[0]
        assert self.S == Nsx * Nsy, f"sensor count mismatch: got {self.S}, expected {Nsx*Nsy}"

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        """
        For each sample:
          1) sample a synthetic current pattern J_true
          2) compute B-field at sensors via Biot–Savart
          3) add Gaussian noise
          4) reshape into (3, Nsx, Nsy)
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        Nsx, Nsy = self.Nsx, self.Nsy
        S = self.S

        # ---------------------------------------------
        # 1) sample current pattern: (3,Nx,Ny,Nz)
        # ---------------------------------------------
        J_true = sample_current_pattern(Nx, Ny, Nz, device=self.device)
        # just sanity check
        assert J_true.shape == (3, Nx, Ny, Nz), \
            f"J_true shape {J_true.shape}, expected (3,{Nx},{Ny},{Nz})"

        # ---------------------------------------------
        # 2) forward Biot–Savart to sensors
        #    forward_biot_savart_sensors:
        #       J: (3,Nx,Ny,Nz) or (B,3,Nx,Ny,Nz)
        #       sensor_coords: (S,3)
        #       → (B,S,3)
        # ---------------------------------------------
        B_sensors = forward_biot_savart_sensors(J_true, self.sensor_coords)
        # if J_true had no batch, we now have (1,S,3)
        if B_sensors.dim() == 3 and B_sensors.size(0) == 1:
            B_sensors = B_sensors[0]  # (S,3)

        # sanity check: (256,3)
        assert B_sensors.shape == (S, 3), \
            f"expected (S,3) with S={S}, got {B_sensors.shape}"

        # ---------------------------------------------
        # 3) add noise in sensor space
        # ---------------------------------------------
        if self.noise_sigma > 0:
            B_sensors = B_sensors + self.noise_sigma * torch.randn_like(B_sensors)

        # ---------------------------------------------
        # 4) reshape into (3,Nsx,Nsy) for the network
        # ---------------------------------------------
        # currently: (S,3) = (Nsx*Nsy,3)
        B_grid = B_sensors.view(Nsx, Nsy, 3).permute(2, 0, 1)  # (3,Nsx,Nsy)

        sample = {
            "B": B_grid,           # (3,Nsx,Nsy)
            "J": J_true,          # (3,Nx,Ny,Nz)
            "X": self.X,          # (1,1,Nx,Ny,Nz)
            "Y": self.Y,
            "Z": self.Z,
            "sensor_coords": self.sensor_coords,  # (S,3)
        }
        return sample
