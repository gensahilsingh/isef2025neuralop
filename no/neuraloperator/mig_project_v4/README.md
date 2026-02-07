# Cardiac Inverse Problem Pipeline

A complete synthetic pipeline for MCG/MIG-like inverse problems with disease classification.

## Overview

This pipeline:
1. Generates synthetic 3D cardiac current density fields (J) for 4 disease classes
2. Computes magnetic field (B) using Biot-Savart operator
3. Adds realistic measurement noise
4. Trains a 3D Fourier Neural Operator (FNO) to reconstruct J from noisy B
5. Trains a 3D CNN classifier to classify disease from reconstructed J
6. Runs stratified k-fold cross-validation with proper nested structure
7. Generates a PDF report with visualizations and metrics

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with defaults (grid=16, n=500, fast training)
python train.py

# Check outputs
ls results/
# pipeline_report.pdf
# volumes/
```

## Command Line Options

### Data Generation
- `--grid`: Grid size (default: 16)
- `--n`: Number of samples (default: 500)
- `--obs_mode`: Observation mode - "full" or "sensors" (default: full)
- `--n_sensors`: Number of sensors if obs_mode=sensors (default: 64)
- `--noise`: Noise level (default: 0.05)

### FNO Settings
- `--fno_modes`: Fourier modes (default: 8)
- `--fno_width`: Hidden width (default: 32)
- `--fno_depth`: Number of layers (default: 4)
- `--fno_dropout`: Dropout rate (default: 0.1)
- `--fno_epochs`: Max epochs (default: 150)
- `--fno_batch`: Batch size (default: 4)
- `--fno_lr`: Learning rate (default: 1e-3)
- `--lambda_phys`: Physics loss weight (default: 0.1)

### Classifier Settings
- `--cls_dropout`: Dropout rate (default: 0.3)
- `--label_smooth`: Label smoothing (default: 0.1)
- `--cls_epochs`: Max epochs (default: 80)
- `--cls_batch`: Batch size (default: 8)
- `--cls_lr`: Learning rate (default: 1e-3)

### Cross-Validation
- `--fast_cv`: Use fast CV mode (single FNO, CV only on classifier)
- `--k_folds`: Number of folds (auto-selected if not specified)

### Other
- `--mc_samples`: MC samples for uncertainty (default: 20)
- `--mc_dropout`: Enable MC dropout for uncertainty
- `--seed`: Random seed (default: 42)
- `--output`: Output directory (default: results)

## Examples

```bash
# Larger grid, more samples
python train.py --grid 32 --n 1000

# Sensor observations (sparse)
python train.py --obs_mode sensors --n_sensors 128

# Fast CV mode (quicker but less rigorous)
python train.py --fast_cv

# Reduce overfitting (more regularization)
python train.py --fno_dropout 0.2 --cls_dropout 0.4 --label_smooth 0.15

# Custom seed for reproducibility
python train.py --seed 123
```

## Output Files

```
results/
├── pipeline_report.pdf    # Multi-page PDF with all visualizations
└── volumes/
    ├── all_volumes.pt     # PyTorch format
    ├── J_true.npy         # True current density
    ├── B_obs.npy          # Observed magnetic field
    ├── J_mean.npy         # Reconstructed mean
    ├── J_std.npy          # Reconstruction uncertainty
    └── labels.npy         # Class labels
```

## Running Tests

```bash
python tests.py
```

## Debugging Tips

### If accuracy drops to chance level (~25%):

1. **Check sanity accuracy**: The pipeline runs a "sanity check" classifier on true J.
   - If sanity acc is low: Generator is producing unlearnable distributions
   - If sanity acc is high but recon acc is low: FNO reconstruction quality is poor

2. **Increase regularization**:
   ```bash
   python train.py --fno_dropout 0.2 --cls_dropout 0.4
   ```

3. **Check for scale mismatch**: Ensure J and B have similar magnitude ranges

4. **Use fast_cv to isolate issues**:
   ```bash
   python train.py --fast_cv
   ```

### If training is unstable:

1. Lower learning rate:
   ```bash
   python train.py --fno_lr 5e-4 --cls_lr 5e-4
   ```

2. Increase physics loss weight:
   ```bash
   python train.py --lambda_phys 0.5
   ```

### If training is too slow:

1. Reduce grid size: `--grid 12`
2. Use fast CV: `--fast_cv`
3. Reduce samples: `--n 200`

## Disease Classes

1. **Normal**: Smooth, coherent current patterns with fiber alignment
2. **Ischemia**: Perfusion deficit with magnitude reduction and altered coherence
3. **Arrhythmia**: Disrupted conduction with vortex cores and increased directional variance
4. **Hypertrophy**: Wall thickening with boundary amplification and fiber enhancement

## Architecture

- **FNO3d**: 3D Fourier Neural Operator with spectral convolutions
- **CNN3DClassifier**: 3D CNN with global average pooling
- **BiotSavartOperator**: FFT-based differentiable forward model

## Citation

If you use this code, please cite appropriately.
