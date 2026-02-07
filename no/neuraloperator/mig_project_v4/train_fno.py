"""FNO training loop with physics-informed loss."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import os

from fno import FNO3d
from biot_savart import BiotSavartOperator
from config import ExperimentConfig


class FNOTrainer:
    """Trainer for FNO inverse model."""
    
    def __init__(
        self,
        model: FNO3d,
        config: ExperimentConfig,
        biot_savart: BiotSavartOperator
    ):
        """
        Initialize trainer.
        
        Args:
            model: FNO model
            config: Experiment configuration
            biot_savart: Forward operator for physics loss
        """
        self.model = model
        self.config = config
        self.biot_savart = biot_savart
        self.device = config.device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.fno_lr,
            weight_decay=config.fno_weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Loss
        self.mse = nn.MSELoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_state = None
    
    def compute_loss(
        self,
        B_input: torch.Tensor,
        J_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute data + physics loss.
        
        Args:
            B_input: Input B field (B, C, X, Y, Z) where C=3 or 4 with mask
            J_true: True current density (B, 3, X, Y, Z)
            mask: Optional sensor mask
            
        Returns:
            total_loss: Combined loss
            metrics: Dict of individual losses
        """
        # Extract mask from input if present
        if B_input.shape[1] == 4:
            mask = B_input[:, 3:4]  # (B, 1, X, Y, Z)
            B_obs = B_input[:, :3]
        else:
            B_obs = B_input
        
        # Forward pass
        J_pred = self.model(B_input)
        
        # Data loss
        data_loss = self.mse(J_pred, J_true)
        
        # Physics loss: B_pred from J_pred should match B_obs
        B_pred = self.biot_savart(J_pred)
        
        if mask is not None:
            # Apply mask for sensor mode
            B_pred_masked = B_pred * mask
            B_obs_masked = B_obs * mask
            physics_loss = self.mse(B_pred_masked, B_obs_masked)
        else:
            physics_loss = self.mse(B_pred, B_obs)
        
        # Total loss
        total_loss = data_loss + self.config.lambda_phys * physics_loss
        
        metrics = {
            'data': data_loss.item(),
            'physics': physics_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, metrics
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        augment: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_metrics = {'data': 0, 'physics': 0, 'total': 0}
        n_batches = 0
        
        for B_input, J_true, _ in train_loader:
            B_input = B_input.to(self.device)
            J_true = J_true.to(self.device)
            
            # Augmentation: add input noise
            if augment and self.config.fno_aug_noise > 0:
                noise = torch.randn_like(B_input[:, :3]) * self.config.fno_aug_noise
                B_input = B_input.clone()
                B_input[:, :3] = B_input[:, :3] + noise
            
            # Forward + backward
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(B_input, J_true)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            for k, v in metrics.items():
                total_metrics[k] += v
            n_batches += 1
        
        return {k: v / n_batches for k, v in total_metrics.items()}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_metrics = {'data': 0, 'physics': 0, 'total': 0}
        n_batches = 0
        
        for B_input, J_true, _ in val_loader:
            B_input = B_input.to(self.device)
            J_true = J_true.to(self.device)
            
            _, metrics = self.compute_loss(B_input, J_true)
            
            for k, v in metrics.items():
                total_metrics[k] += v
            n_batches += 1
        
        return {k: v / n_batches for k, v in total_metrics.items()}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_path: Path to save best checkpoint
            
        Returns:
            history: Training history
        """
        history = {'train_total': [], 'val_total': [], 'lr': []}
        
        for epoch in range(self.config.fno_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_metrics['total'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            print(f"[FNO] Epoch {epoch+1:3d} | "
                  f"train_total={train_metrics['total']:.4f} | "
                  f"val_total={val_metrics['total']:.4f} | "
                  f"lr={current_lr:.2e}")
            
            # History
            history['train_total'].append(train_metrics['total'])
            history['val_total'].append(val_metrics['total'])
            history['lr'].append(current_lr)
            
            # Early stopping
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                if checkpoint_path:
                    torch.save(self.best_state, checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.fno_patience:
                    print(f"[FNO] Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        return history


def train_fno(
    train_dataset,
    val_dataset,
    config: ExperimentConfig,
    checkpoint_dir: Optional[str] = None
) -> Tuple[FNO3d, Dict]:
    """
    Train FNO model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Experiment configuration
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        model: Trained FNO model
        history: Training history
    """
    # Determine input channels
    sample_B, _, _ = train_dataset[0]
    in_channels = sample_B.shape[0]
    
    print(f"[FNO] Input channels: {in_channels}")
    
    # Create model
    model = FNO3d(
        in_channels=in_channels,
        out_channels=3,
        width=config.fno_width,
        modes=config.fno_modes,
        depth=config.fno_depth,
        dropout=config.fno_dropout
    ).to(config.device)
    
    print(f"[FNO] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.fno_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.fno_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create forward operator
    biot_savart = BiotSavartOperator(config.grid_size, config.device)
    
    # Create trainer
    trainer = FNOTrainer(model, config, biot_savart)
    
    # Checkpoint path
    checkpoint_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "fno_best.pt")
    
    # Train
    history = trainer.train(train_loader, val_loader, checkpoint_path)
    
    return model, history


@torch.no_grad()
def reconstruct_all(
    model: FNO3d,
    dataset,
    config: ExperimentConfig,
    batch_size: int = 8
) -> torch.Tensor:
    """
    Reconstruct J for all samples in dataset.
    
    Args:
        model: Trained FNO model
        dataset: Dataset with (B, J, label) samples
        config: Configuration
        batch_size: Batch size for reconstruction
        
    Returns:
        J_recon: Reconstructed currents (N, 3, X, Y, Z)
    """
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    J_recon_list = []
    
    for B_input, _, _ in loader:
        B_input = B_input.to(config.device)
        J_pred = model(B_input)
        J_recon_list.append(J_pred.cpu())
    
    J_recon = torch.cat(J_recon_list, dim=0)
    
    return J_recon
