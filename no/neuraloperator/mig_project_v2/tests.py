r"""
tests.py
========

Sanity tests for:
- biot_savart shape + zero field
- synthetic dataset shapes
- fno forward pass shape
- classifier forward + predict_with_features
"""

import torch

from biot_savart import compute_biot_savart
from synthetic_data import generate_dataset, HeartCurrentDataset
from fno import FNO3d
from cnn_classifier import HeartDiseaseClassifier3D


def test_biot_savart_zero_field() -> None:
    grid_size = 4
    J = torch.zeros(grid_size, grid_size, grid_size, 3, dtype=torch.float32)
    B = compute_biot_savart(J)
    assert B.shape == (grid_size, grid_size, grid_size, 3)
    assert torch.allclose(B, torch.zeros_like(B), atol=1e-6)


def test_generate_dataset_shapes() -> None:
    currents, fields, labels = generate_dataset(dataset_size=3, grid_size=4, noise_level=0.0, seed=123)
    assert len(currents) == len(fields) == len(labels) == 3
    for J, B, y in zip(currents, fields, labels):
        assert J.shape == (4, 4, 4, 3)
        assert B.shape == (4, 4, 4, 3)
        assert isinstance(y, int)


def test_fno_forward_shape() -> None:
    model = FNO3d(in_channels=3, out_channels=3, modes=(2, 2, 2), width=8)
    x = torch.randn(2, 3, 4, 4, 4)
    y = model(x)
    assert y.shape == (2, 3, 4, 4, 4)


def test_classifier_forward_shape() -> None:
    model = HeartDiseaseClassifier3D(in_channels=3, num_classes=4, base_channels=8)
    x = torch.randn(5, 3, 16, 16, 16)
    logits = model(x)
    assert logits.shape == (5, 4)

    probs, feats = model.predict_with_features(x[:1])
    assert probs.shape == (1, 4)
    assert "mean_magnitude" in feats and "std_magnitude" in feats and "max_magnitude_coord" in feats


if __name__ == "__main__":
    test_biot_savart_zero_field()
    test_generate_dataset_shapes()
    test_fno_forward_shape()
    test_classifier_forward_shape()
    print("all tests passed")
