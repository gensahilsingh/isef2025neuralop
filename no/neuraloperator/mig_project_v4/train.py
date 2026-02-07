"""Main orchestrator for the cardiac inverse problem pipeline."""
import torch
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional

from config import ExperimentConfig, parse_args, set_seed
from synthetic_data import generate_dataset, CardiacDataset, ClassifierDataset
from biot_savart import BiotSavartOperator
from fno import FNO3d, mc_inference
from cnn_classifier import CNN3DClassifier, mc_classifier_inference
from train_fno import train_fno, reconstruct_all
from train_classifier import train_classifier, evaluate_classifier
from cv import stratified_split, stratified_kfold, select_k, print_fold_info
from metrics import compute_all_metrics, print_metrics, check_performance_warnings, predictive_entropy
from report import generate_pdf_report


def save_volumes(
    output_dir: str,
    J_true: torch.Tensor,
    B_obs: torch.Tensor,
    J_mean: torch.Tensor,
    J_std: torch.Tensor,
    labels: torch.Tensor
) -> None:
    """Save volumes to disk in both .pt and .npy formats."""
    volumes_dir = os.path.join(output_dir, "volumes")
    os.makedirs(volumes_dir, exist_ok=True)
    
    # Save as PyTorch
    torch.save({
        'J_true': J_true,
        'B_obs': B_obs,
        'J_mean': J_mean,
        'J_std': J_std,
        'labels': labels
    }, os.path.join(volumes_dir, "all_volumes.pt"))
    
    # Save as NumPy
    np.save(os.path.join(volumes_dir, "J_true.npy"), J_true.numpy())
    np.save(os.path.join(volumes_dir, "B_obs.npy"), B_obs.numpy())
    np.save(os.path.join(volumes_dir, "J_mean.npy"), J_mean.numpy())
    np.save(os.path.join(volumes_dir, "J_std.npy"), J_std.numpy())
    np.save(os.path.join(volumes_dir, "labels.npy"), labels.numpy())
    
    print(f"[Save] Volumes saved to {volumes_dir}")


def run_sanity_check(
    J_true: torch.Tensor,
    labels: torch.Tensor,
    config: ExperimentConfig
) -> float:
    """
    Run sanity check: train classifier on true J.
    
    If this fails, generator is producing unlearnable distributions.
    """
    print("\n" + "=" * 50)
    print("SANITY CHECK: Classifier on True J")
    print("=" * 50)
    
    # Convert to channel-first
    J_cf = J_true.permute(0, 4, 1, 2, 3)  # (N, 3, X, Y, Z)
    
    # Quick split
    train_idx, val_idx, _ = stratified_split(labels, train_ratio=0.7, val_ratio=0.15, seed=config.seed)
    
    train_ds = ClassifierDataset(J_cf[train_idx], labels[train_idx], augment=True)
    val_ds = ClassifierDataset(J_cf[val_idx], labels[val_idx], augment=False)
    
    # Quick training (fewer epochs)
    sanity_config = ExperimentConfig(
        **{**config.__dict__, 'cls_epochs': 30, 'cls_patience': 10}
    )
    
    _, _, true_j_acc = train_classifier(train_ds, val_ds, sanity_config)
    
    print(f"\n[Sanity] True J classifier accuracy: {true_j_acc:.4f}")
    
    # Check warnings
    warnings = check_performance_warnings(true_j_acc, len(config.classes))
    for w in warnings:
        print(f"\n⚠️  {w}")
    
    return true_j_acc


def run_nested_cv(
    B_all: torch.Tensor,
    J_all: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor],
    config: ExperimentConfig
) -> Tuple[List[float], Dict, Dict]:
    """
    Run correct nested cross-validation.
    
    For each fold:
    1. Train FNO on fold's training data
    2. Reconstruct J for fold's train and val
    3. Train classifier on reconstructed train
    4. Evaluate on reconstructed val
    """
    print("\n" + "=" * 50)
    print("NESTED CROSS-VALIDATION (Correct Pipeline)")
    print("=" * 50)
    
    # Select k
    k = config.k_folds if config.k_folds else select_k(labels)
    folds = stratified_kfold(labels, k, config.seed)
    print_fold_info(folds, labels)
    
    fold_accuracies = []
    all_preds = []
    all_labels = []
    
    # Store histories from last fold for reporting
    last_fno_history = None
    last_cls_history = None
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{k} ---")
        
        # Create FNO datasets
        train_idx_t = torch.tensor(train_idx)
        val_idx_t = torch.tensor(val_idx)
        
        fno_train_ds = CardiacDataset(
            J_all[train_idx_t], B_all[train_idx_t], labels[train_idx_t],
            mask=mask, augment=True, aug_noise=config.fno_aug_noise
        )
        fno_val_ds = CardiacDataset(
            J_all[val_idx_t], B_all[val_idx_t], labels[val_idx_t],
            mask=mask, augment=False
        )
        
        # Train FNO for this fold
        fno_model, fno_history = train_fno(fno_train_ds, fno_val_ds, config)
        last_fno_history = fno_history
        
        # Reconstruct J for classifier datasets
        J_recon_train = reconstruct_all(fno_model, fno_train_ds, config)
        J_recon_val = reconstruct_all(fno_model, fno_val_ds, config)
        
        # Create classifier datasets
        cls_train_ds = ClassifierDataset(J_recon_train, labels[train_idx_t], augment=True)
        cls_val_ds = ClassifierDataset(J_recon_val, labels[val_idx_t], augment=False)
        
        # Train classifier
        cls_model, cls_history, val_acc = train_classifier(cls_train_ds, cls_val_ds, config)
        last_cls_history = cls_history
        
        fold_accuracies.append(val_acc)
        print(f"[Fold {fold_idx + 1}] Val accuracy: {val_acc:.4f}")
        
        # Get predictions for final metrics
        preds, true_labels = evaluate_classifier(cls_model, cls_val_ds, config)
        all_preds.append(preds)
        all_labels.append(true_labels)
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\n[CV] Final: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Aggregate predictions
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    return fold_accuracies, last_fno_history, last_cls_history


def run_fast_cv(
    B_all: torch.Tensor,
    J_all: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor],
    config: ExperimentConfig
) -> Tuple[List[float], Dict, Dict, FNO3d]:
    """
    Run fast CV mode: single FNO, CV only on classifier.
    """
    print("\n" + "=" * 50)
    print("FAST CV MODE")
    print("=" * 50)
    
    # Split for FNO training
    train_idx, val_idx, test_idx = stratified_split(labels, seed=config.seed)
    train_idx_t = torch.tensor(train_idx)
    val_idx_t = torch.tensor(val_idx)
    
    # Train single FNO
    fno_train_ds = CardiacDataset(
        J_all[train_idx_t], B_all[train_idx_t], labels[train_idx_t],
        mask=mask, augment=True
    )
    fno_val_ds = CardiacDataset(
        J_all[val_idx_t], B_all[val_idx_t], labels[val_idx_t],
        mask=mask, augment=False
    )
    
    fno_model, fno_history = train_fno(fno_train_ds, fno_val_ds, config)
    
    # Reconstruct all training data
    all_train_idx = train_idx + val_idx
    all_train_idx_t = torch.tensor(all_train_idx)
    
    full_train_ds = CardiacDataset(
        J_all[all_train_idx_t], B_all[all_train_idx_t], labels[all_train_idx_t],
        mask=mask, augment=False
    )
    
    J_recon_all = reconstruct_all(fno_model, full_train_ds, config)
    
    # CV on classifier
    k = config.k_folds if config.k_folds else select_k(labels[all_train_idx_t])
    folds = stratified_kfold(labels[all_train_idx_t], k, config.seed)
    
    fold_accuracies = []
    last_cls_history = None
    
    for fold_idx, (train_sub_idx, val_sub_idx) in enumerate(folds):
        print(f"\n--- Classifier Fold {fold_idx + 1}/{k} ---")
        
        train_sub_idx_t = torch.tensor(train_sub_idx)
        val_sub_idx_t = torch.tensor(val_sub_idx)
        
        cls_train_ds = ClassifierDataset(
            J_recon_all[train_sub_idx_t], 
            labels[all_train_idx_t][train_sub_idx_t], 
            augment=True
        )
        cls_val_ds = ClassifierDataset(
            J_recon_all[val_sub_idx_t], 
            labels[all_train_idx_t][val_sub_idx_t], 
            augment=False
        )
        
        _, cls_history, val_acc = train_classifier(cls_train_ds, cls_val_ds, config)
        last_cls_history = cls_history
        
        fold_accuracies.append(val_acc)
        print(f"[Fold {fold_idx + 1}] Val accuracy: {val_acc:.4f}")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\n[Fast CV] Final: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return fold_accuracies, fno_history, last_cls_history, fno_model


def main():
    """Main entry point."""
    # Parse config
    config = parse_args()
    
    # Set seed
    set_seed(config.seed)
    
    print("\n" + "=" * 60)
    print("CARDIAC INVERSE PROBLEM PIPELINE")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Grid size: {config.grid_size}")
    print(f"Samples: {config.n_samples}")
    print(f"Obs mode: {config.obs_mode}")
    print(f"Fast CV: {config.fast_cv}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Generate dataset
    print("\n[1/6] Generating synthetic dataset...")
    J_all, B_all, labels, mask = generate_dataset(
        n_samples=config.n_samples,
        grid_size=config.grid_size,
        noise_level=config.noise_level,
        obs_mode=config.obs_mode,
        n_sensors=config.n_sensors,
        device=config.device,
        classes=config.classes
    )
    
    # Move to CPU for cross-validation (to save GPU memory)
    J_all = J_all.cpu()
    B_all = B_all.cpu()
    labels = labels.cpu()
    if mask is not None:
        mask = mask.cpu()
    
    # Sanity check
    print("\n[2/6] Running sanity check (classifier on true J)...")
    true_j_acc = run_sanity_check(J_all, labels, config)
    
    # Run cross-validation
    print("\n[3/6] Running cross-validation...")
    if config.fast_cv:
        fold_accuracies, fno_history, cls_history, fno_model = run_fast_cv(
            B_all, J_all, labels, mask, config
        )
    else:
        fold_accuracies, fno_history, cls_history = run_nested_cv(
            B_all, J_all, labels, mask, config
        )
        # Train final FNO for uncertainty estimation
        train_idx, val_idx, _ = stratified_split(labels, seed=config.seed)
        fno_train_ds = CardiacDataset(
            J_all[torch.tensor(train_idx)], B_all[torch.tensor(train_idx)], 
            labels[torch.tensor(train_idx)], mask=mask, augment=True
        )
        fno_val_ds = CardiacDataset(
            J_all[torch.tensor(val_idx)], B_all[torch.tensor(val_idx)],
            labels[torch.tensor(val_idx)], mask=mask, augment=False
        )
        fno_model, _ = train_fno(fno_train_ds, fno_val_ds, config)
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    # Check performance warnings
    warnings = check_performance_warnings(mean_acc, len(config.classes), true_j_acc)
    for w in warnings:
        print(f"\n⚠️  {w}")
    
    # Uncertainty estimation on example
    print("\n[4/6] Computing uncertainty estimates...")
    example_idx = 0
    B_example = B_all[example_idx:example_idx+1].permute(0, 4, 1, 2, 3).to(config.device)
    if mask is not None:
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).to(config.device)
        B_example = torch.cat([B_example, mask_expanded], dim=1)
    
    J_mean, J_std, _ = mc_inference(
        fno_model, B_example, 
        n_samples=config.mc_samples,
        input_noise=config.fno_aug_noise,
        use_mc_dropout=config.mc_dropout
    )
    
    # Train final classifier for metrics
    print("\n[5/6] Training final classifier for evaluation...")
    train_idx, val_idx, test_idx = stratified_split(labels, seed=config.seed)
    
    # Use full data for final model
    fno_full_ds = CardiacDataset(
        J_all, B_all, labels, mask=mask, augment=False
    )
    J_recon_all = reconstruct_all(fno_model, fno_full_ds, config)
    
    cls_train_ds = ClassifierDataset(J_recon_all[torch.tensor(train_idx)], labels[torch.tensor(train_idx)], augment=True)
    cls_val_ds = ClassifierDataset(J_recon_all[torch.tensor(val_idx)], labels[torch.tensor(val_idx)], augment=False)
    cls_test_ds = ClassifierDataset(J_recon_all[torch.tensor(test_idx)], labels[torch.tensor(test_idx)], augment=False)
    
    cls_model, _, _ = train_classifier(cls_train_ds, cls_val_ds, config)
    
    # Final evaluation
    preds, true_labels = evaluate_classifier(cls_model, cls_test_ds, config)
    metrics = compute_all_metrics(preds, true_labels, config.classes)
    print_metrics(metrics, config.classes)
    
    # Classifier uncertainty for example
    J_recon_example = J_recon_all[example_idx:example_idx+1].to(config.device)
    mean_probs, std_probs, entropy = mc_classifier_inference(cls_model, J_recon_example, n_samples=20)
    
    # Generate report
    print("\n[6/6] Generating PDF report...")
    
    # Compute magnitude volumes for visualization
    J_true_mag = J_all[example_idx].norm(dim=-1).numpy()
    B_obs_mag = B_all[example_idx].norm(dim=-1).numpy()
    J_mean_mag = J_mean[0].norm(dim=0).cpu().numpy()
    J_std_mag = J_std[0].norm(dim=0).cpu().numpy()
    
    generate_pdf_report(
        output_path=os.path.join(config.output_dir, "pipeline_report.pdf"),
        J_true=J_true_mag,
        B_obs=B_obs_mag,
        J_mean=J_mean_mag,
        J_std=J_std_mag,
        metrics=metrics,
        class_names=config.classes,
        fno_history=fno_history,
        cls_history=cls_history,
        fold_accuracies=fold_accuracies,
        mean_acc=mean_acc,
        std_acc=std_acc,
        mean_probs=mean_probs[0].cpu().numpy(),
        std_probs=std_probs[0].cpu().numpy(),
        entropy=entropy[0].item()
    )
    
    # Save volumes
    save_volumes(
        config.output_dir,
        J_all[example_idx],
        B_all[example_idx],
        J_mean[0].cpu(),
        J_std[0].cpu(),
        labels
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {config.output_dir}")
    print(f"  - pipeline_report.pdf")
    print(f"  - volumes/")
    print(f"\nFinal CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"True J Accuracy:   {true_j_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
