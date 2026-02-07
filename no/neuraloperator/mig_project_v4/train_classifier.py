"""Classifier training loop."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import os

from cnn_classifier import CNN3DClassifier, LabelSmoothingCrossEntropy
from config import ExperimentConfig


class ClassifierTrainer:
    """Trainer for 3D CNN classifier."""
    
    def __init__(
        self,
        model: CNN3DClassifier,
        config: ExperimentConfig,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Classifier model
            config: Experiment configuration
            class_weights: Optional class weights for imbalanced data
        """
        self.model = model
        self.config = config
        self.device = config.device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.cls_lr,
            weight_decay=config.cls_weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=8
        )
        
        # Loss with label smoothing
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=config.label_smoothing,
            weight=class_weights
        )
        
        # Training state
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            loss: Average loss
            acc: Training accuracy
        """
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for J, labels in train_loader:
            J = J.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(J)
            loss = self.criterion(logits, labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * J.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += J.size(0)
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            loss: Validation loss
            acc: Validation accuracy
        """
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for J, labels in val_loader:
            J = J.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(J)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * J.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += J.size(0)
        
        return total_loss / total, correct / total
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Returns:
            history: Training history
        """
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        
        for epoch in range(self.config.cls_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            print(f"[Classifier] Epoch {epoch+1:3d} | "
                  f"loss={train_loss:.4f} | "
                  f"train_acc={train_acc:.4f} | "
                  f"val_acc={val_acc:.4f} | "
                  f"lr={current_lr:.2e}")
            
            # History
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                if checkpoint_path:
                    torch.save(self.best_state, checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.cls_patience:
                    print(f"[Classifier] Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        return history


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.
    
    Args:
        labels: All labels
        num_classes: Number of classes
        
    Returns:
        weights: Class weights (inverse frequency normalized)
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * num_classes
    return weights


def train_classifier(
    train_dataset,
    val_dataset,
    config: ExperimentConfig,
    checkpoint_dir: Optional[str] = None,
    use_class_weights: bool = True
) -> Tuple[CNN3DClassifier, Dict, float]:
    """
    Train classifier model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Experiment configuration
        checkpoint_dir: Directory to save checkpoints
        use_class_weights: Whether to use class weights
        
    Returns:
        model: Trained classifier
        history: Training history
        best_val_acc: Best validation accuracy
    """
    # Create model
    model = CNN3DClassifier(
        in_channels=3,
        num_classes=len(config.classes),
        dropout=config.cls_dropout
    ).to(config.device)
    
    print(f"[Classifier] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Compute class weights
    class_weights = None
    if use_class_weights:
        all_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
        class_weights = compute_class_weights(all_labels, len(config.classes))
        print(f"[Classifier] Class weights: {class_weights.tolist()}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.cls_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.cls_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create trainer
    trainer = ClassifierTrainer(model, config, class_weights)
    
    # Checkpoint path
    checkpoint_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "classifier_best.pt")
    
    # Train
    history = trainer.train(train_loader, val_loader, checkpoint_path)
    
    return model, history, trainer.best_val_acc


@torch.no_grad()
def evaluate_classifier(
    model: CNN3DClassifier,
    dataset,
    config: ExperimentConfig,
    batch_size: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate classifier on dataset.
    
    Args:
        model: Classifier model
        dataset: Dataset
        config: Configuration
        batch_size: Batch size
        
    Returns:
        preds: Predictions (N,)
        labels: True labels (N,)
    """
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    for J, labels in loader:
        J = J.to(config.device)
        logits = model(J)
        preds = logits.argmax(dim=-1)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels)
    
    return torch.cat(all_preds), torch.cat(all_labels)
