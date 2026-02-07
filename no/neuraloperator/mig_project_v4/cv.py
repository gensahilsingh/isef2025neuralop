"""Stratified K-fold cross-validation utilities."""
import torch
from typing import List, Tuple, Optional
import numpy as np
from collections import Counter


def stratified_split(
    labels: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified train/val/test split.
    
    Args:
        labels: Class labels (N,)
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        train_idx, val_idx, test_idx
    """
    np.random.seed(seed)
    
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    n = len(labels_np)
    classes = np.unique(labels_np)
    
    train_idx = []
    val_idx = []
    test_idx = []
    
    for c in classes:
        class_idx = np.where(labels_np == c)[0]
        np.random.shuffle(class_idx)
        
        n_class = len(class_idx)
        n_train = int(n_class * train_ratio)
        n_val = int(n_class * val_ratio)
        
        train_idx.extend(class_idx[:n_train].tolist())
        val_idx.extend(class_idx[n_train:n_train + n_val].tolist())
        test_idx.extend(class_idx[n_train + n_val:].tolist())
    
    return train_idx, val_idx, test_idx


def stratified_kfold(
    labels: torch.Tensor,
    k: int = 5,
    seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Stratified K-fold split.
    
    Args:
        labels: Class labels (N,)
        k: Number of folds
        seed: Random seed
        
    Returns:
        List of (train_idx, val_idx) tuples for each fold
    """
    np.random.seed(seed)
    
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    n = len(labels_np)
    classes = np.unique(labels_np)
    
    # Group indices by class
    class_indices = {c: np.where(labels_np == c)[0].tolist() for c in classes}
    
    # Shuffle within each class
    for c in classes:
        np.random.shuffle(class_indices[c])
    
    # Assign to folds
    fold_indices = [[] for _ in range(k)]
    
    for c in classes:
        idx_list = class_indices[c]
        for i, idx in enumerate(idx_list):
            fold_indices[i % k].append(idx)
    
    # Create train/val splits
    folds = []
    for i in range(k):
        val_idx = fold_indices[i]
        train_idx = []
        for j in range(k):
            if j != i:
                train_idx.extend(fold_indices[j])
        folds.append((train_idx, val_idx))
    
    return folds


def select_k(
    labels: torch.Tensor,
    min_per_class_val: int = 10,
    default_k: int = 5,
    max_k: int = 10
) -> int:
    """
    Automatically select number of folds.
    
    Args:
        labels: Class labels (N,)
        min_per_class_val: Minimum samples per class in validation
        default_k: Default number of folds
        max_k: Maximum folds to consider
        
    Returns:
        k: Selected number of folds
    """
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    
    n = len(labels_np)
    class_counts = Counter(labels_np)
    min_class_count = min(class_counts.values())
    
    # Check if we can support higher k
    if n > 800 and min_class_count >= 200:
        k = min(max_k, min_class_count // min_per_class_val)
    else:
        k = min(default_k, min_class_count // min_per_class_val)
    
    k = max(2, min(k, max_k))  # Ensure k is at least 2
    
    print(f"[CV] Dataset size: {n}, min class count: {min_class_count}, selected k={k}")
    
    return k


def print_fold_info(folds: List[Tuple[List[int], List[int]]], labels: torch.Tensor) -> None:
    """Print fold size and class distribution."""
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    
    print(f"\n[CV] Fold information ({len(folds)} folds):")
    print("-" * 50)
    
    for i, (train_idx, val_idx) in enumerate(folds):
        train_labels = labels_np[train_idx]
        val_labels = labels_np[val_idx]
        
        train_dist = dict(Counter(train_labels))
        val_dist = dict(Counter(val_labels))
        
        print(f"Fold {i+1}: train={len(train_idx)}, val={len(val_idx)}")
        print(f"  Train dist: {train_dist}")
        print(f"  Val dist:   {val_dist}")
    
    print("-" * 50)
