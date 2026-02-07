"""Disease-aware synthetic cardiac current density generator."""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import numpy as np

from biot_savart import BiotSavartOperator, add_measurement_noise, create_sensor_mask, apply_sensor_mask
from config import assert_spatial_last, assert_channel_first


def smooth_3d(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian-like smoothing using average pooling."""
    kernel_size = max(3, int(2 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 27
    
    if x.dim() == 4:  # (C, X, Y, Z)
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # Apply multiple passes of avg pooling for approximation
    for _ in range(int(sigma)):
        x = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
    
    if squeeze:
        x = x.squeeze(0)
    return x


def create_base_current(
    grid_size: int,
    device: str = "cpu",
    smoothness: float = 2.0
) -> torch.Tensor:
    """
    Create base smooth divergence-free current density field.
    
    Uses vector potential + curl approach for divergence-free field.
    
    Args:
        grid_size: Size of 3D grid
        device: Target device
        smoothness: Smoothing factor
        
    Returns:
        J: Current density (grid_size, grid_size, grid_size, 3) spatial-last
    """
    n = grid_size
    
    # Create random vector potential A
    A = torch.randn(3, n, n, n, device=device)
    
    # Smooth the vector potential
    A = smooth_3d(A, smoothness)
    
    # Compute curl(A) = J using finite differences
    # curl(A)_x = dA_z/dy - dA_y/dz
    # curl(A)_y = dA_x/dz - dA_z/dx
    # curl(A)_z = dA_y/dx - dA_x/dy
    
    def diff_axis(x, axis):
        """Central difference along axis."""
        return torch.roll(x, -1, axis) - torch.roll(x, 1, axis)
    
    J_x = diff_axis(A[2], 1) - diff_axis(A[1], 2)  # dAz/dy - dAy/dz
    J_y = diff_axis(A[0], 2) - diff_axis(A[2], 0)  # dAx/dz - dAz/dx
    J_z = diff_axis(A[1], 0) - diff_axis(A[0], 1)  # dAy/dx - dAx/dy
    
    J = torch.stack([J_x, J_y, J_z], dim=-1)  # (X, Y, Z, 3)
    
    # Normalize to reasonable magnitude
    J = J / (J.norm(dim=-1, keepdim=True).mean() + 1e-8)
    
    return J


def create_fiber_field(grid_size: int, device: str = "cpu") -> torch.Tensor:
    """
    Create synthetic fiber direction field (simplified helical structure).
    
    Args:
        grid_size: Size of 3D grid
        device: Target device
        
    Returns:
        Fiber directions (grid_size, grid_size, grid_size, 3)
    """
    n = grid_size
    
    # Create coordinate grids
    x = torch.linspace(-1, 1, n, device=device)
    y = torch.linspace(-1, 1, n, device=device)
    z = torch.linspace(-1, 1, n, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Helical angle varies with depth (z)
    theta = np.pi * Z  # Rotation angle
    
    # Fiber direction (roughly circumferential with helical twist)
    r = torch.sqrt(X**2 + Y**2) + 1e-6
    fx = -Y / r * torch.cos(theta) - Z * torch.sin(theta) * X / r
    fy = X / r * torch.cos(theta) - Z * torch.sin(theta) * Y / r
    fz = torch.sin(theta) * 0.3
    
    fiber = torch.stack([fx, fy, fz], dim=-1)
    fiber = fiber / (fiber.norm(dim=-1, keepdim=True) + 1e-8)
    
    return fiber


def create_ellipsoid_region(
    grid_size: int,
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
    noise_amount: float = 0.1,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Create smooth ellipsoid region with noisy boundary.
    
    Args:
        grid_size: Size of 3D grid
        center: Center (cx, cy, cz) in normalized [-1, 1] coords
        radii: Radii (rx, ry, rz) in normalized coords
        noise_amount: Boundary noise level
        device: Target device
        
    Returns:
        Smooth mask (grid_size, grid_size, grid_size) with values [0, 1]
    """
    n = grid_size
    x = torch.linspace(-1, 1, n, device=device)
    y = torch.linspace(-1, 1, n, device=device)
    z = torch.linspace(-1, 1, n, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    cx, cy, cz = center
    rx, ry, rz = radii
    
    # Ellipsoid distance
    dist = ((X - cx) / rx)**2 + ((Y - cy) / ry)**2 + ((Z - cz) / rz)**2
    
    # Add boundary noise
    if noise_amount > 0:
        noise = torch.randn(n, n, n, device=device) * noise_amount
        noise = smooth_3d(noise.unsqueeze(0), sigma=1.5).squeeze(0)
        dist = dist + noise
    
    # Smooth transition (sigmoid-like)
    mask = torch.sigmoid(5 * (1 - dist))
    
    return mask


def generate_normal(grid_size: int, device: str = "cpu") -> torch.Tensor:
    """Generate normal cardiac current pattern."""
    J = create_base_current(grid_size, device, smoothness=2.0)
    
    # Add subtle fiber alignment
    fiber = create_fiber_field(grid_size, device)
    fiber_component = (J * fiber).sum(dim=-1, keepdim=True) * fiber
    J = 0.7 * J + 0.3 * fiber_component
    
    # Normalize
    J = J / (J.norm(dim=-1, keepdim=True).mean() + 1e-8)
    
    return J


def generate_ischemia(grid_size: int, device: str = "cpu") -> torch.Tensor:
    """
    Generate ischemic current pattern with perfusion deficit.
    
    Features:
    - Irregular ellipsoid deficit region
    - Magnitude reduction in deficit
    - Altered directional coherence
    - Smooth border zone
    """
    J_base = generate_normal(grid_size, device)
    
    # Create deficit region (subendocardial style - closer to center)
    center = (
        np.random.uniform(-0.3, 0.3),
        np.random.uniform(-0.3, 0.3),
        np.random.uniform(-0.5, 0.5)
    )
    radii = (
        np.random.uniform(0.2, 0.4),
        np.random.uniform(0.15, 0.35),
        np.random.uniform(0.15, 0.35)
    )
    
    deficit_mask = create_ellipsoid_region(grid_size, center, radii, noise_amount=0.15, device=device)
    deficit_mask = deficit_mask.unsqueeze(-1)  # (X, Y, Z, 1)
    
    # Magnitude reduction (60-85% reduction in deficit) - STRONGER
    reduction = np.random.uniform(0.6, 0.85)
    J_reduced = J_base * (1 - deficit_mask * reduction)
    
    # Directional perturbation in deficit - STRONGER
    random_dir = torch.randn(grid_size, grid_size, grid_size, 3, device=device)
    random_dir = smooth_3d(random_dir.permute(3, 0, 1, 2), sigma=1.0).permute(1, 2, 3, 0)
    random_dir = random_dir / (random_dir.norm(dim=-1, keepdim=True) + 1e-8)
    
    J_perturbed = J_reduced + deficit_mask * 0.5 * random_dir * J_base.norm(dim=-1, keepdim=True)
    
    # Keep lower overall magnitude (characteristic of ischemia)
    J_perturbed = J_perturbed / (J_perturbed.norm(dim=-1, keepdim=True).mean() + 1e-8) * 0.7
    
    return J_perturbed


def generate_arrhythmia(grid_size: int, device: str = "cpu") -> torch.Tensor:
    """
    Generate arrhythmic current pattern with disrupted conduction.
    
    Features:
    - Local vortex/reentry-like cores
    - Increased directional variance
    - Preserved smoothness and magnitude
    """
    J_base = generate_normal(grid_size, device)
    
    n = grid_size
    x = torch.linspace(-1, 1, n, device=device)
    y = torch.linspace(-1, 1, n, device=device)
    z = torch.linspace(-1, 1, n, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Create 1-3 vortex cores
    n_vortices = np.random.randint(1, 4)
    vortex_field = torch.zeros(n, n, n, 3, device=device)
    
    for _ in range(n_vortices):
        # Random vortex center
        cx = np.random.uniform(-0.5, 0.5)
        cy = np.random.uniform(-0.5, 0.5)
        cz = np.random.uniform(-0.5, 0.5)
        
        # Distance from vortex axis (z-aligned for simplicity)
        r = torch.sqrt((X - cx)**2 + (Y - cy)**2) + 1e-6
        
        # Vortex influence (decays with distance)
        influence = torch.exp(-2 * r**2).unsqueeze(-1)
        
        # Vortex direction (tangential)
        vx = -(Y - cy) / r
        vy = (X - cx) / r
        vz = torch.zeros_like(vx)
        
        # Random rotation to vary vortex axis
        angle = np.random.uniform(0, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        vx_rot = vx * cos_a - vz * sin_a
        vz_rot = vx * sin_a + vz * cos_a
        
        vortex_dir = torch.stack([vx_rot, vy, vz_rot], dim=-1)
        vortex_field = vortex_field + influence * vortex_dir
    
    # Add local phase variations
    phase_noise = torch.randn(n, n, n, device=device) * 0.5
    phase_noise = smooth_3d(phase_noise.unsqueeze(0), sigma=1.5).squeeze(0)
    
    # Apply vortex perturbation - STRONGER
    vortex_strength = np.random.uniform(0.7, 1.2)
    J = J_base + vortex_strength * vortex_field * J_base.norm(dim=-1, keepdim=True)
    
    # Add direction variance via random rotations - STRONGER
    angle_perturb = phase_noise.unsqueeze(-1) * 0.6
    cos_p = torch.cos(angle_perturb)
    sin_p = torch.sin(angle_perturb)
    
    # Simple rotation around z-axis
    J_rot_x = J[..., 0] * cos_p.squeeze(-1) - J[..., 1] * sin_p.squeeze(-1)
    J_rot_y = J[..., 0] * sin_p.squeeze(-1) + J[..., 1] * cos_p.squeeze(-1)
    J = torch.stack([J_rot_x, J_rot_y, J[..., 2]], dim=-1)
    
    # Normalize to similar magnitude as normal
    J = J / (J.norm(dim=-1, keepdim=True).mean() + 1e-8)
    
    return J


def generate_hypertrophy(grid_size: int, device: str = "cpu") -> torch.Tensor:
    """
    Generate hypertrophic current pattern.
    
    Features:
    - Boundary shell amplification (wall thickening)
    - Mild global amplification
    - Fiber-direction enhancement
    """
    J_base = generate_normal(grid_size, device)
    fiber = create_fiber_field(grid_size, device)
    
    n = grid_size
    x = torch.linspace(-1, 1, n, device=device)
    y = torch.linspace(-1, 1, n, device=device)
    z = torch.linspace(-1, 1, n, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Create shell region (between two ellipsoids)
    r = torch.sqrt(X**2 + Y**2 + (Z * 1.5)**2)
    
    # Shell mask: high at boundary, low at center and outside
    inner_r = 0.3 + np.random.uniform(-0.1, 0.1)
    outer_r = 0.7 + np.random.uniform(-0.1, 0.1)
    shell = torch.sigmoid(8 * (r - inner_r)) * torch.sigmoid(8 * (outer_r - r))
    shell = shell.unsqueeze(-1)  # (X, Y, Z, 1)
    
    # Shell amplification - STRONGER
    shell_amp = np.random.uniform(1.8, 2.5)
    J = J_base * (1 + shell * (shell_amp - 1))
    
    # Fiber-direction amplification - STRONGER
    fiber_component = (J * fiber).sum(dim=-1, keepdim=True) * fiber
    fiber_amp = np.random.uniform(1.3, 1.8)
    J = J + shell * (fiber_amp - 1) * fiber_component
    
    # Global amplification - STRONGER
    global_amp = np.random.uniform(1.2, 1.5)
    J = J * global_amp
    
    # Normalize to higher magnitude (characteristic of hypertrophy)
    target_mag = np.random.uniform(1.4, 1.8)
    J = J / (J.norm(dim=-1, keepdim=True).mean() + 1e-8) * target_mag
    
    return J


GENERATORS = {
    "normal": generate_normal,
    "ischemia": generate_ischemia,
    "arrhythmia": generate_arrhythmia,
    "hypertrophy": generate_hypertrophy,
}


def generate_sample(
    label: int,
    grid_size: int,
    device: str = "cpu",
    classes: List[str] = ["normal", "ischemia", "arrhythmia", "hypertrophy"]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a single sample (J, B_clean).
    
    Args:
        label: Class label index
        grid_size: Size of 3D grid
        device: Target device
        classes: List of class names
        
    Returns:
        J: Current density (X, Y, Z, 3) spatial-last
        B: Magnetic field (X, Y, Z, 3) spatial-last
    """
    class_name = classes[label]
    generator = GENERATORS[class_name]
    
    J = generator(grid_size, device)
    assert_spatial_last(J, 3, "J")
    
    # Compute B from J
    biot_savart = BiotSavartOperator(grid_size, device)
    J_cf = J.permute(3, 0, 1, 2)  # (3, X, Y, Z)
    B_cf = biot_savart(J_cf)
    B = B_cf.permute(1, 2, 3, 0)  # (X, Y, Z, 3)
    
    return J, B


def generate_dataset(
    n_samples: int,
    grid_size: int,
    noise_level: float = 0.05,
    obs_mode: str = "full",
    n_sensors: int = 64,
    device: str = "cpu",
    classes: List[str] = ["normal", "ischemia", "arrhythmia", "hypertrophy"]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate full dataset.
    
    Args:
        n_samples: Total number of samples
        grid_size: Size of 3D grid
        noise_level: Noise level for B
        obs_mode: "full" or "sensors"
        n_sensors: Number of sensors if obs_mode="sensors"
        device: Target device
        classes: List of class names
        
    Returns:
        J_all: All current densities (N, X, Y, Z, 3)
        B_all: All noisy magnetic fields (N, X, Y, Z, 3) or (N, X, Y, Z, 4) with mask
        labels: Class labels (N,)
        mask: Sensor mask (X, Y, Z) if obs_mode="sensors", else None
    """
    n_classes = len(classes)
    samples_per_class = n_samples // n_classes
    
    J_list = []
    B_list = []
    label_list = []
    
    # Create sensor mask if needed
    mask = None
    if obs_mode == "sensors":
        mask, _ = create_sensor_mask(grid_size, n_sensors, device)
        mask = mask.squeeze(0)  # (X, Y, Z)
    
    print(f"Generating {n_samples} samples ({samples_per_class} per class)...")
    
    for label in range(n_classes):
        for i in range(samples_per_class):
            J, B = generate_sample(label, grid_size, device, classes)
            
            # Add noise to B
            B_noisy = add_measurement_noise(
                B.permute(3, 0, 1, 2),  # to channel-first
                noise_level=noise_level,
                correlated=True
            ).permute(1, 2, 3, 0)  # back to spatial-last
            
            # Apply sensor mask if needed
            if obs_mode == "sensors":
                B_noisy = B_noisy * mask.unsqueeze(-1)
            
            J_list.append(J)
            B_list.append(B_noisy)
            label_list.append(label)
    
    # Stack all
    J_all = torch.stack(J_list, dim=0)  # (N, X, Y, Z, 3)
    B_all = torch.stack(B_list, dim=0)  # (N, X, Y, Z, 3)
    labels = torch.tensor(label_list, device=device)
    
    print(f"Generated dataset: J {J_all.shape}, B {B_all.shape}, labels {labels.shape}")
    
    return J_all, B_all, labels, mask


class CardiacDataset(Dataset):
    """
    PyTorch Dataset for cardiac data.
    
    Returns model-ready tensors in channel-first format.
    """
    
    def __init__(
        self,
        J: torch.Tensor,
        B: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        augment: bool = False,
        aug_noise: float = 0.02,
        aug_shift: int = 2
    ):
        """
        Initialize dataset.
        
        Args:
            J: Current densities (N, X, Y, Z, 3) spatial-last
            B: Magnetic fields (N, X, Y, Z, 3) spatial-last
            labels: Class labels (N,)
            mask: Optional sensor mask (X, Y, Z)
            augment: Whether to apply augmentation
            aug_noise: Augmentation noise level
            aug_shift: Max random shift for augmentation
        """
        self.J = J
        self.B = B
        self.labels = labels
        self.mask = mask
        self.augment = augment
        self.aug_noise = aug_noise
        self.aug_shift = aug_shift
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            B_input: (C, X, Y, Z) where C=3 or C=4 with mask
            J_target: (3, X, Y, Z)
            label: scalar
        """
        J = self.J[idx]  # (X, Y, Z, 3)
        B = self.B[idx]  # (X, Y, Z, 3)
        label = self.labels[idx]
        
        # Convert to channel-first
        J_cf = J.permute(3, 0, 1, 2)  # (3, X, Y, Z)
        B_cf = B.permute(3, 0, 1, 2)  # (3, X, Y, Z)
        
        # Apply augmentation
        if self.augment:
            # Add noise
            if self.aug_noise > 0:
                noise = torch.randn_like(B_cf) * self.aug_noise * B_cf.std()
                B_cf = B_cf + noise
            
            # Random shift
            if self.aug_shift > 0:
                shift = [np.random.randint(-self.aug_shift, self.aug_shift + 1) for _ in range(3)]
                B_cf = torch.roll(B_cf, shifts=shift, dims=(1, 2, 3))
                J_cf = torch.roll(J_cf, shifts=shift, dims=(1, 2, 3))
        
        # Add mask channel if available
        if self.mask is not None:
            mask_expanded = self.mask.unsqueeze(0)  # (1, X, Y, Z)
            B_cf = torch.cat([B_cf, mask_expanded], dim=0)  # (4, X, Y, Z)
        
        return B_cf, J_cf, label


class ClassifierDataset(Dataset):
    """Dataset for classifier training on reconstructed J."""
    
    def __init__(
        self,
        J_recon: torch.Tensor,
        labels: torch.Tensor,
        augment: bool = False
    ):
        """
        Initialize classifier dataset.
        
        Args:
            J_recon: Reconstructed currents (N, 3, X, Y, Z) channel-first
            labels: Class labels (N,)
            augment: Whether to apply augmentation
        """
        self.J = J_recon
        self.labels = labels
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample (J, label)."""
        J = self.J[idx]
        label = self.labels[idx]
        
        if self.augment:
            # Random flip
            for dim in range(1, 4):
                if np.random.random() < 0.5:
                    J = torch.flip(J, dims=[dim])
            
            # Mild noise
            noise = torch.randn_like(J) * 0.02 * J.std()
            J = J + noise
            
            # Mild intensity scaling
            scale = np.random.uniform(0.9, 1.1)
            J = J * scale
        
        return J, label
