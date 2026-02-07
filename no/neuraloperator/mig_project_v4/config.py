"""Configuration and CLI argument parsing for the cardiac inverse pipeline."""
from dataclasses import dataclass, field
from typing import Optional, List
import argparse
import random
import numpy as np
import torch


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    # Data generation
    grid_size: int = 16
    n_samples: int = 500
    obs_mode: str = "full"  # "full" or "sensors"
    n_sensors: int = 64
    noise_level: float = 0.05
    
    # FNO architecture
    fno_modes: int = 8
    fno_width: int = 32
    fno_depth: int = 4
    fno_dropout: float = 0.1
    
    # FNO training
    fno_epochs: int = 150
    fno_batch_size: int = 4
    fno_lr: float = 1e-3
    fno_weight_decay: float = 1e-4
    fno_patience: int = 20
    lambda_phys: float = 0.1
    fno_aug_noise: float = 0.02
    fno_aug_shift: int = 2
    
    # Classifier architecture
    cls_dropout: float = 0.3
    label_smoothing: float = 0.1
    
    # Classifier training
    cls_epochs: int = 80
    cls_batch_size: int = 8
    cls_lr: float = 1e-3
    cls_weight_decay: float = 1e-4
    cls_patience: int = 15
    
    # Cross-validation
    fast_cv: bool = False
    k_folds: Optional[int] = None  # Auto-select if None
    
    # Uncertainty
    mc_samples: int = 20
    mc_dropout: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Paths
    output_dir: str = "results"
    
    # Device
    device: str = "auto"
    
    # Disease classes
    classes: List[str] = field(default_factory=lambda: ["normal", "ischemia", "arrhythmia", "hypertrophy"])
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> ExperimentConfig:
    """Parse command line arguments and return config."""
    parser = argparse.ArgumentParser(description="Cardiac Inverse Problem Pipeline")
    
    # Data generation
    parser.add_argument("--grid", type=int, default=16, help="Grid size (default: 16)")
    parser.add_argument("--n", type=int, default=500, help="Number of samples (default: 500)")
    parser.add_argument("--obs_mode", type=str, default="full", choices=["full", "sensors"],
                        help="Observation mode (default: full)")
    parser.add_argument("--n_sensors", type=int, default=64, help="Number of sensors if obs_mode=sensors")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level (default: 0.05)")
    
    # FNO
    parser.add_argument("--fno_modes", type=int, default=8, help="FNO Fourier modes")
    parser.add_argument("--fno_width", type=int, default=32, help="FNO width")
    parser.add_argument("--fno_depth", type=int, default=4, help="FNO depth")
    parser.add_argument("--fno_dropout", type=float, default=0.1, help="FNO dropout")
    parser.add_argument("--fno_epochs", type=int, default=150, help="FNO epochs")
    parser.add_argument("--fno_batch", type=int, default=4, help="FNO batch size")
    parser.add_argument("--fno_lr", type=float, default=1e-3, help="FNO learning rate")
    parser.add_argument("--lambda_phys", type=float, default=0.1, help="Physics loss weight")
    
    # Classifier
    parser.add_argument("--cls_dropout", type=float, default=0.3, help="Classifier dropout")
    parser.add_argument("--label_smooth", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--cls_epochs", type=int, default=80, help="Classifier epochs")
    parser.add_argument("--cls_batch", type=int, default=8, help="Classifier batch size")
    parser.add_argument("--cls_lr", type=float, default=1e-3, help="Classifier learning rate")
    
    # CV
    parser.add_argument("--fast_cv", action="store_true", help="Use fast CV mode")
    parser.add_argument("--k_folds", type=int, default=None, help="Number of CV folds (auto if None)")
    
    # Uncertainty
    parser.add_argument("--mc_samples", type=int, default=20, help="MC samples for uncertainty")
    parser.add_argument("--mc_dropout", action="store_true", help="Enable MC dropout")
    
    # General
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        grid_size=args.grid,
        n_samples=args.n,
        obs_mode=args.obs_mode,
        n_sensors=args.n_sensors,
        noise_level=args.noise,
        fno_modes=args.fno_modes,
        fno_width=args.fno_width,
        fno_depth=args.fno_depth,
        fno_dropout=args.fno_dropout,
        fno_epochs=args.fno_epochs,
        fno_batch_size=args.fno_batch,
        fno_lr=args.fno_lr,
        lambda_phys=args.lambda_phys,
        cls_dropout=args.cls_dropout,
        label_smoothing=args.label_smooth,
        cls_epochs=args.cls_epochs,
        cls_batch_size=args.cls_batch,
        cls_lr=args.cls_lr,
        fast_cv=args.fast_cv,
        k_folds=args.k_folds,
        mc_samples=args.mc_samples,
        mc_dropout=args.mc_dropout,
        seed=args.seed,
        output_dir=args.output,
    )
    
    return config


# Shape utilities
def cf_to_last(x: torch.Tensor) -> torch.Tensor:
    """Convert channel-first (B,C,X,Y,Z) to channel-last (B,X,Y,Z,C)."""
    assert x.dim() == 5, f"Expected 5D tensor, got {x.dim()}D"
    return x.permute(0, 2, 3, 4, 1)


def last_to_cf(x: torch.Tensor) -> torch.Tensor:
    """Convert channel-last (B,X,Y,Z,C) to channel-first (B,C,X,Y,Z)."""
    assert x.dim() == 5, f"Expected 5D tensor, got {x.dim()}D"
    return x.permute(0, 4, 1, 2, 3)


def assert_shape(x: torch.Tensor, expected: tuple, name: str = "tensor") -> None:
    """Assert tensor has expected shape."""
    if x.shape != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {x.shape}")


def assert_spatial_last(x: torch.Tensor, channels: int, name: str = "tensor") -> None:
    """Assert tensor is spatial-last format with given channels."""
    assert x.dim() == 4, f"{name}: expected 4D (X,Y,Z,C), got {x.dim()}D"
    assert x.shape[-1] == channels, f"{name}: expected {channels} channels, got {x.shape[-1]}"


def assert_channel_first(x: torch.Tensor, channels: int, name: str = "tensor") -> None:
    """Assert tensor is channel-first format with given channels."""
    if x.dim() == 4:
        assert x.shape[0] == channels, f"{name}: expected {channels} channels, got {x.shape[0]}"
    elif x.dim() == 5:
        assert x.shape[1] == channels, f"{name}: expected {channels} channels, got {x.shape[1]}"
    else:
        raise ValueError(f"{name}: expected 4D or 5D tensor, got {x.dim()}D")
