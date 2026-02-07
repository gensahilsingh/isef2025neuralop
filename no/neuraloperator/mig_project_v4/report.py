"""PDF report generation with research-style visualizations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import os


def plot_central_slices(
    volumes: Dict[str, np.ndarray],
    titles: Dict[str, str],
    slice_idx: Optional[int] = None
) -> plt.Figure:
    """
    Plot central slices of multiple volumes.
    
    Args:
        volumes: Dict of volume name -> (X, Y, Z) array
        titles: Dict of volume name -> title string
        slice_idx: Slice index (default: center)
        
    Returns:
        fig: Matplotlib figure
    """
    n_vols = len(volumes)
    fig, axes = plt.subplots(1, n_vols, figsize=(4 * n_vols, 4))
    
    if n_vols == 1:
        axes = [axes]
    
    for ax, (name, vol) in zip(axes, volumes.items()):
        if slice_idx is None:
            idx = vol.shape[2] // 2
        else:
            idx = slice_idx
        
        im = ax.imshow(vol[:, :, idx].T, origin='lower', cmap='viridis')
        ax.set_title(titles.get(name, name))
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def plot_multislice_grid(
    volume: np.ndarray,
    title: str,
    n_slices: int = 6,
    axis: int = 2
) -> plt.Figure:
    """
    Plot multiple slices of a volume.
    
    Args:
        volume: 3D array (X, Y, Z)
        title: Figure title
        n_slices: Number of slices to show
        axis: Axis along which to slice (0, 1, or 2)
        
    Returns:
        fig: Matplotlib figure
    """
    n_total = volume.shape[axis]
    slice_indices = np.linspace(0, n_total - 1, n_slices, dtype=int)
    
    ncols = 3
    nrows = (n_slices + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()
    
    for i, idx in enumerate(slice_indices):
        if axis == 0:
            slice_data = volume[idx, :, :]
        elif axis == 1:
            slice_data = volume[:, idx, :]
        else:
            slice_data = volume[:, :, idx]
        
        im = axes[i].imshow(slice_data.T, origin='lower', cmap='viridis')
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Hide unused axes
    for i in range(n_slices, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_mip(
    volume: np.ndarray,
    title: str
) -> plt.Figure:
    """
    Plot Maximum Intensity Projection.
    
    Args:
        volume: 3D array (X, Y, Z)
        title: Figure title
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    mip_x = volume.max(axis=0)
    mip_y = volume.max(axis=1)
    mip_z = volume.max(axis=2)
    
    for ax, mip, label in zip(axes, [mip_x, mip_y, mip_z], ['X', 'Y', 'Z']):
        im = ax.imshow(mip.T, origin='lower', cmap='hot')
        ax.set_title(f'MIP along {label}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str]
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix (n_classes, n_classes)
        class_names: List of class names
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(cm, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', 
                          color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.tight_layout()
    return fig


def plot_uncertainty_bars(
    mean_probs: np.ndarray,
    std_probs: np.ndarray,
    class_names: List[str],
    entropy: float
) -> plt.Figure:
    """
    Plot classifier uncertainty as bar plot.
    
    Args:
        mean_probs: Mean class probabilities
        std_probs: Std of class probabilities
        class_names: List of class names
        entropy: Predictive entropy
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(class_names))
    
    ax.bar(x, mean_probs, yerr=std_probs * 1.96, capsize=5, color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.set_title(f'Classifier Uncertainty (Entropy: {entropy:.3f})')
    
    plt.tight_layout()
    return fig


def plot_training_history(
    fno_history: Dict,
    cls_history: Dict
) -> plt.Figure:
    """
    Plot training history for FNO and classifier.
    
    Args:
        fno_history: FNO training history
        cls_history: Classifier training history
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # FNO loss
    ax = axes[0, 0]
    ax.plot(fno_history['train_total'], label='Train')
    ax.plot(fno_history['val_total'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('FNO Training Loss')
    ax.legend()
    ax.set_yscale('log')
    
    # FNO learning rate
    ax = axes[0, 1]
    ax.plot(fno_history['lr'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('FNO Learning Rate')
    ax.set_yscale('log')
    
    # Classifier accuracy
    ax = axes[1, 0]
    ax.plot(cls_history['train_acc'], label='Train')
    ax.plot(cls_history['val_acc'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classifier Accuracy')
    ax.legend()
    
    # Classifier loss
    ax = axes[1, 1]
    ax.plot(cls_history['train_loss'], label='Train')
    ax.plot(cls_history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Classifier Loss')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_cv_results(
    fold_accuracies: List[float],
    mean_acc: float,
    std_acc: float
) -> plt.Figure:
    """
    Plot cross-validation results.
    
    Args:
        fold_accuracies: List of per-fold accuracies
        mean_acc: Mean accuracy
        std_acc: Std of accuracy
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(fold_accuracies))
    
    ax.bar(x, fold_accuracies, color='steelblue', alpha=0.8)
    ax.axhline(mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.3f} ± {std_acc:.3f}')
    ax.fill_between([-0.5, len(fold_accuracies) - 0.5], 
                     mean_acc - std_acc, mean_acc + std_acc,
                     alpha=0.2, color='red')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in x])
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross-Validation Results')
    ax.legend()
    
    plt.tight_layout()
    return fig


def generate_pdf_report(
    output_path: str,
    J_true: np.ndarray,
    B_obs: np.ndarray,
    J_mean: np.ndarray,
    J_std: np.ndarray,
    metrics: Dict,
    class_names: List[str],
    fno_history: Dict,
    cls_history: Dict,
    fold_accuracies: List[float],
    mean_acc: float,
    std_acc: float,
    mean_probs: Optional[np.ndarray] = None,
    std_probs: Optional[np.ndarray] = None,
    entropy: Optional[float] = None
) -> None:
    """
    Generate full PDF report.
    
    Args:
        output_path: Path to save PDF
        J_true: True current density magnitude (X, Y, Z)
        B_obs: Observed B magnitude (X, Y, Z)
        J_mean: Mean reconstructed J magnitude (X, Y, Z)
        J_std: Std of reconstructed J (X, Y, Z)
        metrics: Classification metrics dict
        class_names: List of class names
        fno_history: FNO training history
        cls_history: Classifier training history
        fold_accuracies: Per-fold accuracies
        mean_acc: Mean CV accuracy
        std_acc: Std of CV accuracy
        mean_probs: Mean class probabilities for example
        std_probs: Std of class probabilities
        entropy: Predictive entropy
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # Page 1: Overview
        fig = plot_central_slices(
            {'J_true': J_true, 'B_obs': B_obs, 'J_mean': J_mean, 'J_std': J_std},
            {'J_true': 'True |J|', 'B_obs': 'Noisy |B|', 'J_mean': 'Recon Mean |J|', 'J_std': 'Recon Std |J|'}
        )
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 2-4: Multislice grids
        for name, vol, title in [
            ('J_true', J_true, 'True |J| - Axial Slices'),
            ('J_mean', J_mean, 'Reconstructed Mean |J| - Axial Slices'),
            ('J_std', J_std, 'Reconstruction Uncertainty - Axial Slices')
        ]:
            fig = plot_multislice_grid(vol, title, n_slices=6, axis=2)
            pdf.savefig(fig)
            plt.close(fig)
        
        # Page 5-7: MIP
        for vol, title in [
            (J_true, 'True |J| - MIP'),
            (J_mean, 'Reconstructed Mean |J| - MIP'),
            (J_std, 'Reconstruction Uncertainty - MIP')
        ]:
            fig = plot_mip(vol, title)
            pdf.savefig(fig)
            plt.close(fig)
        
        # Page 8: Credible interval width
        ci_width = 2 * 1.96 * J_std
        fig = plot_multislice_grid(ci_width, '95% Credible Interval Width', n_slices=6, axis=2)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 9: Confusion matrix
        fig = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 10: Classifier uncertainty
        if mean_probs is not None and std_probs is not None and entropy is not None:
            fig = plot_uncertainty_bars(mean_probs, std_probs, class_names, entropy)
            pdf.savefig(fig)
            plt.close(fig)
        
        # Page 11: Training history
        fig = plot_training_history(fno_history, cls_history)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 12: CV results
        fig = plot_cv_results(fold_accuracies, mean_acc, std_acc)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 13: Metrics summary
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        text = f"""
CLASSIFICATION RESULTS SUMMARY
==============================

Overall Accuracy: {metrics['accuracy']:.4f}
Macro F1 Score:   {metrics['macro_f1']:.4f}

Cross-Validation: {mean_acc:.4f} ± {std_acc:.4f}

Per-Class Performance:
"""
        for name in class_names:
            m = metrics['per_class'][name]
            text += f"\n  {name}:\n"
            text += f"    Precision: {m['precision']:.4f}\n"
            text += f"    Recall:    {m['recall']:.4f}\n"
            text += f"    F1:        {m['f1']:.4f}"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"[Report] Saved PDF report to {output_path}")
