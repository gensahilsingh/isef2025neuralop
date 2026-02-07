"""Minimal tests for pipeline verification."""
import torch
import numpy as np
import sys


def test_shapes():
    """Test tensor shape conventions."""
    print("Testing shape conventions...")
    
    from config import cf_to_last, last_to_cf, assert_shape
    
    # Test cf_to_last
    x_cf = torch.randn(2, 3, 16, 16, 16)
    x_last = cf_to_last(x_cf)
    assert x_last.shape == (2, 16, 16, 16, 3), f"cf_to_last failed: {x_last.shape}"
    
    # Test last_to_cf
    x_back = last_to_cf(x_last)
    assert x_back.shape == x_cf.shape, f"last_to_cf failed: {x_back.shape}"
    
    # Test assert_shape
    assert_shape(x_cf, (2, 3, 16, 16, 16), "test")
    
    print("  ✓ Shape tests passed")


def test_biot_savart():
    """Test Biot-Savart operator."""
    print("Testing Biot-Savart...")
    
    from biot_savart import BiotSavartOperator
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    op = BiotSavartOperator(16, device)
    
    # Zero input
    J_zero = torch.zeros(1, 3, 16, 16, 16, device=device)
    B_zero = op(J_zero)
    assert torch.allclose(B_zero, torch.zeros_like(B_zero), atol=1e-6)
    
    # Non-zero finite output
    J_rand = torch.randn(2, 3, 16, 16, 16, device=device)
    B_rand = op(J_rand)
    assert torch.isfinite(B_rand).all()
    assert B_rand.abs().max() > 1e-10
    
    print("  ✓ Biot-Savart tests passed")


def test_synthetic_data():
    """Test synthetic data generation."""
    print("Testing synthetic data generation...")
    
    from synthetic_data import generate_dataset, CardiacDataset
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    J, B, labels, mask = generate_dataset(
        n_samples=40,
        grid_size=16,
        noise_level=0.05,
        obs_mode="full",
        device=device
    )
    
    # Check shapes
    assert J.shape == (40, 16, 16, 16, 3), f"J shape: {J.shape}"
    assert B.shape == (40, 16, 16, 16, 3), f"B shape: {B.shape}"
    assert labels.shape == (40,), f"labels shape: {labels.shape}"
    
    # Check label distribution
    assert (labels == 0).sum() == 10
    assert (labels == 1).sum() == 10
    assert (labels == 2).sum() == 10
    assert (labels == 3).sum() == 10
    
    # Test Dataset
    ds = CardiacDataset(J.cpu(), B.cpu(), labels.cpu())
    B_sample, J_sample, label = ds[0]
    assert B_sample.shape == (3, 16, 16, 16)
    assert J_sample.shape == (3, 16, 16, 16)
    
    print("  ✓ Synthetic data tests passed")


def test_fno():
    """Test FNO model."""
    print("Testing FNO...")
    
    from fno import FNO3d, mc_inference
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = FNO3d(in_channels=3, out_channels=3, width=16, modes=4, depth=2).to(device)
    
    # Forward pass
    x = torch.randn(2, 3, 16, 16, 16, device=device)
    y = model(x)
    
    assert y.shape == x.shape, f"Output shape: {y.shape}"
    assert torch.isfinite(y).all()
    
    # MC inference
    mean, std, samples = mc_inference(model, x, n_samples=3)
    assert mean.shape == x.shape
    assert std.shape == x.shape
    
    print("  ✓ FNO tests passed")


def test_classifier():
    """Test CNN classifier."""
    print("Testing classifier...")
    
    from cnn_classifier import CNN3DClassifier, mc_classifier_inference
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CNN3DClassifier(in_channels=3, num_classes=4, base_channels=16).to(device)
    
    # Forward pass
    x = torch.randn(4, 3, 16, 16, 16, device=device)
    logits = model(x)
    
    assert logits.shape == (4, 4), f"Output shape: {logits.shape}"
    assert torch.isfinite(logits).all()
    
    # MC inference
    mean_probs, std_probs, entropy = mc_classifier_inference(model, x, n_samples=3)
    assert mean_probs.shape == (4, 4)
    assert entropy.shape == (4,)
    
    print("  ✓ Classifier tests passed")


def test_cv():
    """Test cross-validation utilities."""
    print("Testing CV utilities...")
    
    from cv import stratified_split, stratified_kfold, select_k
    
    labels = torch.tensor([0]*30 + [1]*30 + [2]*30 + [3]*30)
    
    # Stratified split
    train, val, test = stratified_split(labels)
    assert len(train) + len(val) + len(test) == 120
    
    # Check stratification
    train_labels = labels[train]
    for c in range(4):
        assert (train_labels == c).sum() > 0
    
    # K-fold
    k = select_k(labels)
    folds = stratified_kfold(labels, k)
    assert len(folds) == k
    
    # Check all indices covered
    all_val = []
    for train_idx, val_idx in folds:
        all_val.extend(val_idx)
    assert len(set(all_val)) == 120
    
    print("  ✓ CV tests passed")


def test_metrics():
    """Test metrics computation."""
    print("Testing metrics...")
    
    from metrics import accuracy, confusion_matrix, precision_recall_f1, compute_all_metrics
    
    preds = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    labels = torch.tensor([0, 1, 2, 3, 1, 1, 2, 0])
    
    acc = accuracy(preds, labels)
    assert 0 <= acc <= 1
    
    cm = confusion_matrix(preds, labels, 4)
    assert cm.shape == (4, 4)
    assert cm.sum() == 8
    
    prec, rec, f1 = precision_recall_f1(cm)
    assert len(prec) == 4
    
    metrics = compute_all_metrics(preds, labels, ["a", "b", "c", "d"])
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    
    print("  ✓ Metrics tests passed")


def test_no_leakage():
    """Test that CV splits don't leak."""
    print("Testing for data leakage...")
    
    from cv import stratified_kfold
    
    labels = torch.tensor([0]*25 + [1]*25 + [2]*25 + [3]*25)
    
    folds = stratified_kfold(labels, k=5)
    
    for i, (train_i, val_i) in enumerate(folds):
        # Check no overlap
        train_set = set(train_i)
        val_set = set(val_i)
        
        assert len(train_set & val_set) == 0, f"Leakage in fold {i}"
        
        # Check all data used
        assert len(train_set) + len(val_set) == 100
    
    print("  ✓ No leakage detected")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("RUNNING PIPELINE TESTS")
    print("=" * 50 + "\n")
    
    tests = [
        test_shapes,
        test_biot_savart,
        test_synthetic_data,
        test_fno,
        test_classifier,
        test_cv,
        test_metrics,
        test_no_leakage,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
