"""Evaluation metrics for classification."""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy."""
    return (preds == labels).float().mean().item()


def confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        preds: Predictions (N,)
        labels: True labels (N,)
        num_classes: Number of classes
        
    Returns:
        cm: Confusion matrix (num_classes, num_classes)
             cm[i,j] = count of true=i, pred=j
    """
    preds_np = preds.numpy() if isinstance(preds, torch.Tensor) else preds
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for t, p in zip(labels_np, preds_np):
        cm[t, p] += 1
    
    return cm


def precision_recall_f1(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-class precision, recall, F1.
    
    Args:
        cm: Confusion matrix
        
    Returns:
        precision: Per-class precision
        recall: Per-class recall
        f1: Per-class F1 score
    """
    num_classes = cm.shape[0]
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp + 1e-8)
        recall[i] = tp / (tp + fn + 1e-8)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)
    
    return precision, recall, f1


def compute_all_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str]
) -> Dict:
    """
    Compute all classification metrics.
    
    Args:
        preds: Predictions (N,)
        labels: True labels (N,)
        class_names: List of class names
        
    Returns:
        metrics: Dictionary of all metrics
    """
    num_classes = len(class_names)
    
    acc = accuracy(preds, labels)
    cm = confusion_matrix(preds, labels, num_classes)
    precision, recall, f1 = precision_recall_f1(cm)
    
    metrics = {
        'accuracy': acc,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': f1.mean(),
        'per_class': {}
    }
    
    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        }
    
    return metrics


def print_metrics(metrics: Dict, class_names: List[str]) -> None:
    """Print metrics in readable format."""
    print("\n" + "=" * 50)
    print("CLASSIFICATION METRICS")
    print("=" * 50)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    print("\nPer-class metrics:")
    print("-" * 50)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 50)
    
    for name in class_names:
        m = metrics['per_class'][name]
        print(f"{name:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    
    print("-" * 50)
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    header = "        " + " ".join([f"{name[:8]:>8}" for name in class_names])
    print(header)
    for i, name in enumerate(class_names):
        row = f"{name[:8]:<8}" + " ".join([f"{cm[i,j]:>8d}" for j in range(len(class_names))])
        print(row)
    
    print("=" * 50)


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute predictive entropy.
    
    Args:
        probs: Class probabilities (N, C) or (C,)
        
    Returns:
        entropy: Predictive entropy (N,) or scalar
    """
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error.
    
    Args:
        probs: Class probabilities (N, C)
        labels: True labels (N,)
        n_bins: Number of bins
        
    Returns:
        ece: Expected calibration error
    """
    probs_np = probs.numpy() if isinstance(probs, torch.Tensor) else probs
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    confidences = probs_np.max(axis=1)
    predictions = probs_np.argmax(axis=1)
    accuracies = (predictions == labels_np).astype(float)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)
    
    return ece


def check_performance_warnings(
    accuracy: float,
    num_classes: int,
    true_j_accuracy: Optional[float] = None
) -> List[str]:
    """
    Check for performance warnings.
    
    Args:
        accuracy: Validation accuracy
        num_classes: Number of classes
        true_j_accuracy: Accuracy on true J (if available)
        
    Returns:
        warnings: List of warning messages
    """
    warnings = []
    
    chance = 1.0 / num_classes
    
    if accuracy < chance + 0.05:
        warnings.append(
            f"WARNING: Near-chance performance ({accuracy:.2%} vs chance {chance:.2%}). "
            "Likely generator issue or label mismatch."
        )
    
    if true_j_accuracy is not None:
        if true_j_accuracy > 0.7 and accuracy < true_j_accuracy - 0.2:
            warnings.append(
                f"WARNING: True J accuracy ({true_j_accuracy:.2%}) much higher than "
                f"recon J ({accuracy:.2%}). Inverse model quality insufficient or domain shift."
            )
        
        if true_j_accuracy < chance + 0.1:
            warnings.append(
                f"WARNING: True J accuracy ({true_j_accuracy:.2%}) near chance. "
                "Generator produces unlearnable distributions."
            )
    
    return warnings
