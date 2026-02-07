"""3D Fourier Neural Operator for inverse reconstruction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectralConv3d(nn.Module):
    """3D Spectral convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Initialize spectral convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes to keep
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.scale = 1.0 / (in_channels * out_channels)
        
        # Complex weights for each quadrant
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat)
        )
    
    def compl_mul3d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space."""
        # x: (B, in_channels, modes, modes, modes)
        # weights: (in_channels, out_channels, modes, modes, modes)
        return torch.einsum("bixyz,ioxyz->boxyz", x, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, X, Y, Z)
            
        Returns:
            Output tensor (B, C_out, X, Y, Z)
        """
        batch_size = x.shape[0]
        size_x, size_y, size_z = x.shape[2], x.shape[3], x.shape[4]
        
        # FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        
        # Output tensor
        out_ft = torch.zeros(
            batch_size, self.out_channels, size_x, size_y, size_z // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Clamp modes to available frequencies
        m = min(self.modes, size_x // 2, size_y // 2, size_z // 2 + 1)
        
        # Multiply in each quadrant (using clamped modes)
        out_ft[:, :, :m, :m, :m] = self.compl_mul3d(
            x_ft[:, :, :m, :m, :m], self.weights1[:, :, :m, :m, :m]
        )
        out_ft[:, :, -m:, :m, :m] = self.compl_mul3d(
            x_ft[:, :, -m:, :m, :m], self.weights2[:, :, :m, :m, :m]
        )
        out_ft[:, :, :m, -m:, :m] = self.compl_mul3d(
            x_ft[:, :, :m, -m:, :m], self.weights3[:, :, :m, :m, :m]
        )
        out_ft[:, :, -m:, -m:, :m] = self.compl_mul3d(
            x_ft[:, :, -m:, -m:, :m], self.weights4[:, :, :m, :m, :m]
        )
        
        # Inverse FFT
        x = torch.fft.irfftn(out_ft, s=(size_x, size_y, size_z), dim=(-3, -2, -1))
        
        return x


class FNOBlock(nn.Module):
    """Single FNO block with spectral convolution and skip connection."""
    
    def __init__(self, width: int, modes: int, dropout: float = 0.0):
        """
        Initialize FNO block.
        
        Args:
            width: Hidden dimension
            modes: Number of Fourier modes
            dropout: Dropout probability
        """
        super().__init__()
        self.spectral_conv = SpectralConv3d(width, width, modes)
        self.linear = nn.Conv3d(width, width, 1)
        self.norm = nn.InstanceNorm3d(width)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        x1 = self.spectral_conv(x)
        x2 = self.linear(x)
        x = x1 + x2
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator for inverse mapping B -> J.
    
    Architecture:
    - Lifting layer: in_channels -> width
    - N FNO blocks
    - Projection layers: width -> out_channels
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 32,
        modes: int = 8,
        depth: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize FNO3d.
        
        Args:
            in_channels: Input channels (3 for B, 4 with mask)
            out_channels: Output channels (3 for J)
            width: Hidden dimension
            modes: Number of Fourier modes to keep
            depth: Number of FNO blocks
            dropout: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.depth = depth
        
        # Lifting
        self.lift = nn.Conv3d(in_channels, width, 1)
        
        # FNO blocks
        self.blocks = nn.ModuleList([
            FNOBlock(width, modes, dropout) for _ in range(depth)
        ])
        
        # Projection
        self.proj1 = nn.Conv3d(width, width, 1)
        self.proj2 = nn.Conv3d(width, out_channels, 1)
        
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (B, in_channels, X, Y, Z)
            
        Returns:
            Output (B, 3, X, Y, Z)
        """
        assert x.dim() == 5, f"Expected 5D input (B,C,X,Y,Z), got {x.dim()}D"
        assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {x.shape[1]}"
        
        # Lift
        x = self.lift(x)
        
        # FNO blocks
        for block in self.blocks:
            x = block(x)
        
        # Project
        x = self.proj1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.proj2(x)
        
        return x
    
    def enable_dropout(self):
        """Enable dropout for MC inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout3d):
                m.train()
    
    def disable_dropout(self):
        """Disable dropout (normal eval mode)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout3d):
                m.eval()


def mc_inference(
    model: FNO3d,
    x: torch.Tensor,
    n_samples: int = 20,
    input_noise: float = 0.02,
    use_mc_dropout: bool = True
) -> tuple:
    """
    Monte Carlo inference for uncertainty estimation.
    
    Args:
        model: FNO model
        x: Input tensor (B, C, X, Y, Z)
        n_samples: Number of MC samples
        input_noise: Noise level for input perturbation
        use_mc_dropout: Whether to use MC dropout
        
    Returns:
        mean: Mean prediction (B, 3, X, Y, Z)
        std: Standard deviation (B, 3, X, Y, Z)
        samples: All samples (n_samples, B, 3, X, Y, Z)
    """
    model.eval()
    
    if use_mc_dropout:
        model.enable_dropout()
    
    samples = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Add input perturbation
            x_perturbed = x + torch.randn_like(x) * input_noise * x.std()
            
            # Forward pass
            y = model(x_perturbed)
            samples.append(y)
    
    if use_mc_dropout:
        model.disable_dropout()
    
    samples = torch.stack(samples, dim=0)  # (n_samples, B, 3, X, Y, Z)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    
    return mean, std, samples


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    print("Testing FNO3d...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = FNO3d(in_channels=3, out_channels=3, width=32, modes=8, depth=4).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 3, 16, 16, 16, device=device)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    
    assert y.shape == (2, 3, 16, 16, 16), f"Shape mismatch: {y.shape}"
    assert torch.isfinite(y).all(), "Non-finite output"
    print("FNO3d test passed!")
