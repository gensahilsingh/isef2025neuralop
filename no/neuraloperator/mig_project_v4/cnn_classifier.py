"""3D CNN classifier for disease classification from reconstructed J."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class ConvBlock(nn.Module):
    """3D Convolution block with BN, ReLU, and optional pooling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(2) if pool else nn.Identity()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CNN3DClassifier(nn.Module):
    """
    3D CNN classifier for cardiac disease classification.
    
    Architecture:
    - 4 conv blocks with increasing channels
    - Global average pooling
    - FC layers with dropout
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 32,
        dropout: float = 0.3
    ):
        """
        Initialize classifier.
        
        Args:
            in_channels: Number of input channels (3 for J)
            num_classes: Number of disease classes
            base_channels: Base channel count
            dropout: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        c = base_channels
        
        # Convolutional blocks
        self.conv1 = ConvBlock(in_channels, c, dropout=0)
        self.conv2 = ConvBlock(c, c * 2, dropout=dropout * 0.5)
        self.conv3 = ConvBlock(c * 2, c * 4, dropout=dropout)
        self.conv4 = ConvBlock(c * 4, c * 8, pool=False, dropout=dropout)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # FC layers
        self.fc1 = nn.Linear(c * 8, c * 4)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(c * 4, num_classes)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification.
        
        Args:
            x: Input (B, C, X, Y, Z)
            
        Returns:
            Features (B, feature_dim)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (B, C, X, Y, Z)
            
        Returns:
            Logits (B, num_classes)
        """
        assert x.dim() == 5, f"Expected 5D input, got {x.dim()}D"
        assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {x.shape[1]}"
        
        features = self.extract_features(x)
        x = self.fc_dropout(features)
        logits = self.fc2(x)
        
        return logits
    
    def predict_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with feature extraction.
        
        Args:
            x: Input (B, C, X, Y, Z)
            
        Returns:
            probs: Class probabilities (B, num_classes)
            features: Intermediate features (B, feature_dim)
        """
        features = self.extract_features(x)
        x = self.fc_dropout(features)
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)
        
        return probs, features
    
    def enable_dropout(self):
        """Enable dropout for MC inference."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()
    
    def disable_dropout(self):
        """Disable dropout."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.eval()


def mc_classifier_inference(
    model: CNN3DClassifier,
    x: torch.Tensor,
    n_samples: int = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Monte Carlo dropout inference for classifier.
    
    Args:
        model: Classifier model
        x: Input tensor (B, C, X, Y, Z)
        n_samples: Number of MC samples
        
    Returns:
        mean_probs: Mean probabilities (B, num_classes)
        std_probs: Std of probabilities (B, num_classes)
        entropy: Predictive entropy (B,)
    """
    model.eval()
    model.enable_dropout()
    
    prob_samples = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            prob_samples.append(probs)
    
    model.disable_dropout()
    
    prob_samples = torch.stack(prob_samples, dim=0)  # (n_samples, B, C)
    mean_probs = prob_samples.mean(dim=0)
    std_probs = prob_samples.std(dim=0)
    
    # Predictive entropy
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
    
    return mean_probs, std_probs, entropy


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        """
        Initialize.
        
        Args:
            smoothing: Label smoothing factor
            weight: Class weights for imbalanced data
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            pred: Logits (B, C)
            target: Labels (B,)
            
        Returns:
            Loss scalar
        """
        n_classes = pred.size(-1)
        
        # Create smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        log_probs = F.log_softmax(pred, dim=-1)
        
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            weight_expanded = weight[target].unsqueeze(1)
            loss = -(true_dist * log_probs * weight_expanded).sum(dim=-1)
        else:
            loss = -(true_dist * log_probs).sum(dim=-1)
        
        return loss.mean()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    print("Testing CNN3DClassifier...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CNN3DClassifier(in_channels=3, num_classes=4, dropout=0.3).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(4, 3, 16, 16, 16, device=device)
    logits = model(x)
    print(f"Input: {x.shape}, Output: {logits.shape}")
    
    assert logits.shape == (4, 4), f"Shape mismatch: {logits.shape}"
    assert torch.isfinite(logits).all(), "Non-finite output"
    
    # Test predict_with_features
    probs, features = model.predict_with_features(x)
    print(f"Probs: {probs.shape}, Features: {features.shape}")
    
    # Test MC inference
    mean_probs, std_probs, entropy = mc_classifier_inference(model, x, n_samples=5)
    print(f"MC - Mean probs: {mean_probs.shape}, Entropy: {entropy.shape}")
    
    print("CNN3DClassifier test passed!")
