r"""
train.py
========

end-to-end pipeline:
1) generate synthetic data (J, B_noisy, label)
2) train fno: B -> J with data loss + biot–savart physics loss
3) bayesian-style uncertainty via monte carlo input perturbations:
      mean_J, std_J (voxel-wise)
4) train 3d cnn on reconstructed J
5) generate a multi-page pdf report including:
   - multi-slice grids (axial/coronal/sagittal)
   - 3d voxel renderings (thresholded voxels)
   - uncertainty volume visualization + credible bands
   - confusion matrix (classifier grid)
   - classifier mc-dropout uncertainty (mean/std/entropy)

outputs:
- results/pipeline_report.pdf  (multi-page)
- results/volumes/*.pt and *.npy  (true/mean/std volumes)
"""

from __future__ import annotations

import os
import random
from typing import Tuple, List, Dict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from synthetic_data import generate_dataset, HeartCurrentDataset
from fno import FNO3d
from cnn_classifier import HeartDiseaseClassifier3D
from biot_savart import compute_biot_savart


# -----------------------------
# helpers: saving volumes
# -----------------------------
def save_volume(out_dir: str, name: str, t: torch.Tensor) -> None:
    """
    saves both .pt and .npy
    t is expected cpu tensor
    """
    os.makedirs(out_dir, exist_ok=True)
    pt_path = os.path.join(out_dir, f"{name}.pt")
    npy_path = os.path.join(out_dir, f"{name}.npy")
    torch.save(t, pt_path)
    np.save(npy_path, t.numpy())


# -----------------------------
# helpers: plotting
# -----------------------------
def _vol_to_mag(vol_cf: torch.Tensor) -> np.ndarray:
    """
    vol_cf: (3, X, Y, Z) cpu
    returns magnitude (X,Y,Z) numpy
    """
    mag = torch.sqrt((vol_cf ** 2).sum(dim=0) + 1e-12)
    return mag.cpu().numpy()


def plot_mips(mag: np.ndarray, title: str) -> plt.Figure:
    """
    max intensity projections along each axis
    mag: (X,Y,Z)
    """
    mip_xy = mag.max(axis=2)
    mip_xz = mag.max(axis=1)
    mip_yz = mag.max(axis=0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axs:
        ax.axis("off")

    axs[0].imshow(mip_xy)
    axs[0].set_title(f"{title}\nmip xy")

    axs[1].imshow(mip_xz)
    axs[1].set_title(f"{title}\nmip xz")

    axs[2].imshow(mip_yz)
    axs[2].set_title(f"{title}\nmip yz")

    plt.tight_layout()
    return fig


def plot_multislice_grid(mag: np.ndarray, title: str, n_slices: int = 5) -> plt.Figure:
    """
    shows axial/coronal/sagittal slices in a grid
    mag: (X,Y,Z)
    """
    X, Y, Z = mag.shape
    xs = np.linspace(0, X - 1, n_slices, dtype=int)
    ys = np.linspace(0, Y - 1, n_slices, dtype=int)
    zs = np.linspace(0, Z - 1, n_slices, dtype=int)

    fig, axs = plt.subplots(3, n_slices, figsize=(3 * n_slices, 8))

    # axial: fixed z -> mag[:,:,z]
    for i, z in enumerate(zs):
        ax = axs[0, i]
        ax.imshow(mag[:, :, z])
        ax.set_title(f"axial z={z}")
        ax.axis("off")

    # coronal: fixed y -> mag[:,y,:]
    for i, y in enumerate(ys):
        ax = axs[1, i]
        ax.imshow(mag[:, y, :])
        ax.set_title(f"coronal y={y}")
        ax.axis("off")

    # sagittal: fixed x -> mag[x,:,:]
    for i, x in enumerate(xs):
        ax = axs[2, i]
        ax.imshow(mag[x, :, :])
        ax.set_title(f"sagittal x={x}")
        ax.axis("off")

    fig.suptitle(title, y=0.98, fontsize=14)
    plt.tight_layout()
    return fig


def plot_voxel_render(mag: np.ndarray, title: str, quantile: float = 0.995) -> plt.Figure:
    """
    matplotlib 3d voxel render (thresholded).
    uses a high quantile threshold to keep it clean and fast.
    """
    thr = float(np.quantile(mag, quantile))
    filled = mag >= thr

    # to avoid huge 3d renders, downsample if needed
    max_dim = max(mag.shape)
    step = 1
    if max_dim > 40:
        step = 2
    filled_ds = filled[::step, ::step, ::step]
    mag_ds = mag[::step, ::step, ::step]

    # build colors with alpha
    vals = mag_ds[filled_ds]
    if vals.size == 0:
        vals = np.array([thr], dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max() + 1e-12)

    normed = (mag_ds - vmin) / (vmax - vmin + 1e-12)
    cmap = plt.cm.viridis
    colors = cmap(normed)
    colors[..., 3] = 0.25  # alpha

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled_ds, facecolors=colors, edgecolor=None)
    ax.set_title(f"{title}\nvoxels >= q{quantile:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    return fig


def plot_prob_uncertainty(mean_probs: np.ndarray, std_probs: np.ndarray, class_names: List[str], title: str) -> plt.Figure:
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, mean_probs)
    ax.errorbar(x, mean_probs, yerr=std_probs, fmt="none", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# -----------------------------
# training
# -----------------------------
def train_fno(
    model: FNO3d,
    dataloader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    lambda_phys: float = 0.5,
) -> None:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for B_noisy, J_true in dataloader:
            B_noisy = B_noisy.to(device).float().contiguous()
            J_true = J_true.to(device).float().contiguous()

            optimizer.zero_grad(set_to_none=True)

            J_pred = model(B_noisy)  # (B,3,X,Y,Z)
            loss_data = criterion(J_pred, J_true)

            # physics loss via biot–savart: J_pred -> B_pred
            J_pred_spatial = J_pred.permute(0, 2, 3, 4, 1).contiguous()  # (B,X,Y,Z,3)
            B_pred_spatial = compute_biot_savart(J_pred_spatial)         # (B,X,Y,Z,3)
            B_pred_cf = B_pred_spatial.permute(0, 4, 1, 2, 3).contiguous()

            loss_phys = criterion(B_pred_cf, B_noisy)
            loss = loss_data + lambda_phys * loss_phys

            loss.backward()
            optimizer.step()
            running += loss.item() * B_noisy.size(0)

        train_loss = running / len(dataloader.dataset)

        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for B_noisy, J_true in val_loader:
                B_noisy = B_noisy.to(device).float().contiguous()
                J_true = J_true.to(device).float().contiguous()
                J_pred = model(B_noisy)
                val_running += criterion(J_pred, J_true).item() * B_noisy.size(0)
            val_loss = val_running / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")



def estimate_uncertainty(
    model: FNO3d,
    B_sample_cf: torch.Tensor,
    num_samples: int = 16,
    noise_level: float = 0.02,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    monte carlo (input perturbation):
    run multiple forward passes with noisy B and compute voxel-wise mean/std

    returns mean_J, std_J as (3,X,Y,Z) cpu float32
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    B = B_sample_cf.unsqueeze(0).to(device).float().contiguous()  # (1,3,X,Y,Z)

    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            noise = torch.randn_like(B) * float(noise_level)
            J_pred = model(B + noise).squeeze(0)  # (3,X,Y,Z)
            preds.append(J_pred.detach().cpu())

    stack = torch.stack(preds, dim=0)  # (S,3,X,Y,Z)
    mean = stack.mean(dim=0).float()
    std = stack.std(dim=0).float()
    return mean, std


def enable_dropout_only(model: nn.Module) -> None:
    """
    enables dropout at inference for mc dropout while keeping batchnorm frozen.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_classifier(
    classifier: HeartDiseaseClassifier3D,
    x: torch.Tensor,
    T: int = 30,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    x: (1,3,X,Y,Z) on correct device
    returns:
      mean_probs (C,), std_probs (C,), predictive_entropy (scalar)
    """
    enable_dropout_only(classifier)
    probs_list = []
    for _ in range(T):
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        probs_list.append(probs.detach().cpu().numpy())

    P = np.stack(probs_list, axis=0)  # (T,C)
    mean_probs = P.mean(axis=0)
    std_probs = P.std(axis=0)

    # predictive entropy of mean distribution
    p = np.clip(mean_probs, 1e-9, 1.0)
    entropy = float(-(p * np.log(p)).sum())
    return mean_probs, std_probs, entropy


def train_classifier(
    classifier: HeartDiseaseClassifier3D,
    dataloader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int = 5,
    learning_rate: float = 1e-3,
) -> None:
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        correct = 0
        total = 0

        for J_pred, label in dataloader:
            J_pred = J_pred.to(device).float().contiguous()
            label = label.to(device).long().view(-1)

            # harden label range
            label = torch.clamp(label, 0, num_classes - 1)

            optimizer.zero_grad(set_to_none=True)
            logits = classifier(J_pred)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        train_acc = correct / max(1, total)

        classifier.eval()
        with torch.no_grad():
            correct_val = 0
            total_val = 0
            for J_pred, label in val_loader:
                J_pred = J_pred.to(device).float().contiguous()
                label = label.to(device).long().view(-1)
                label = torch.clamp(label, 0, num_classes - 1)
                logits = classifier(J_pred)
                pred = logits.argmax(dim=1)
                correct_val += (pred == label).sum().item()
                total_val += label.size(0)
            val_acc = correct_val / max(1, total_val)

        print(f"Classifier Epoch {epoch+1}/{epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")


@torch.no_grad()
def eval_confusion_matrix(
    classifier: HeartDiseaseClassifier3D,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    classifier.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for x, y in dataloader:
        x = x.to(device).float().contiguous()
        y = torch.clamp(y.to(device).long().view(-1), 0, num_classes - 1)
        logits = classifier(x)
        pred = logits.argmax(dim=1)
        for ti, pi in zip(y.cpu().numpy().tolist(), pred.cpu().numpy().tolist()):
            cm[int(ti), int(pi)] += 1
    return cm


# -----------------------------
# main
# -----------------------------
def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # scale these up once everything is working
    grid_size = 18
    dataset_size = 2000
    val_fraction = 0.2
    batch_size = 2
    fno_epochs = 150
    classifier_epochs = 50

    diseases = ["normal", "ischemia", "arrhythmia", "hypertrophy"]
    num_classes = len(diseases)

    currents, fields, labels = generate_dataset(
        dataset_size=dataset_size,
        grid_size=grid_size,
        diseases=diseases,
        noise_level=0.02,
        seed=42,
    )

    fno_dataset = HeartCurrentDataset(currents, fields)
    val_size = int(val_fraction * len(fno_dataset))
    train_size = len(fno_dataset) - val_size

    fno_train_dataset, fno_val_dataset = random_split(
        fno_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    fno_train_loader = DataLoader(fno_train_dataset, batch_size=batch_size, shuffle=True)
    fno_val_loader = DataLoader(fno_val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    fno_model = FNO3d(in_channels=3, out_channels=3, modes=(6, 6, 6), width=24)
    print("Training FNO...")
    train_fno(
        fno_model,
        fno_train_loader,
        fno_val_loader,
        device,
        epochs=fno_epochs,
        learning_rate=1e-3,
        lambda_phys=0.5,
    )

    # build predicted currents for classifier training
    fno_model.eval()
    J_preds: List[torch.Tensor] = []
    with torch.no_grad():
        for B_noisy, _ in DataLoader(fno_dataset, batch_size=1, shuffle=False):
            B_noisy = B_noisy.to(device).float().contiguous()
            J_pred = fno_model(B_noisy).squeeze(0).detach().cpu()  # (3,X,Y,Z)
            J_preds.append(J_pred)

    class DatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, J_list: List[torch.Tensor], labels_list: List[int]):
            self.J_list = J_list
            self.labels_list = labels_list

        def __len__(self) -> int:
            return len(self.J_list)

        def __getitem__(self, idx: int):
            J = self.J_list[idx].float().contiguous()
            y = int(self.labels_list[idx])
            y = max(0, min(y, num_classes - 1))
            return J, torch.tensor(y, dtype=torch.long)

    classifier_dataset = DatasetWrapper(J_preds, labels)

    val_size_cls = int(val_fraction * len(classifier_dataset))
    train_size_cls = len(classifier_dataset) - val_size_cls

    cls_train_dataset, cls_val_dataset = random_split(
        classifier_dataset,
        [train_size_cls, val_size_cls],
        generator=torch.Generator().manual_seed(1),
    )

    cls_train_loader = DataLoader(cls_train_dataset, batch_size=batch_size, shuffle=True)
    cls_val_loader = DataLoader(cls_val_dataset, batch_size=batch_size, shuffle=False)

    classifier_model = HeartDiseaseClassifier3D(in_channels=3, num_classes=num_classes, base_channels=16, dropout_prob=0.3)
    print("Training classifier...")
    train_classifier(
        classifier_model,
        cls_train_loader,
        cls_val_loader,
        device,
        num_classes=num_classes,
        epochs=classifier_epochs,
        learning_rate=1e-3,
    )

    # confusion matrix on val
    cm = eval_confusion_matrix(classifier_model, cls_val_loader, device, num_classes)

    # pick a sample and compute fno uncertainty volumes
    sample_idx = random.randint(0, dataset_size - 1)
    B_sample_cf = fields[sample_idx].permute(3, 0, 1, 2).contiguous().float()
    J_true_cf = currents[sample_idx].permute(3, 0, 1, 2).contiguous().float()

    mean_J, std_J = estimate_uncertainty(fno_model, B_sample_cf, num_samples=20, noise_level=0.02, device=device)

    # credible bands (voxel-wise): mean ± 1.96 std
    lo_J = (mean_J - 1.96 * std_J).float()
    hi_J = (mean_J + 1.96 * std_J).float()

    # classifier inference on mean_J + mc dropout uncertainty
    x_cls = mean_J.unsqueeze(0).to(device).float().contiguous()
    probs, features = classifier_model.predict_with_features(x_cls)
    pred_class = int(torch.argmax(probs, dim=1).item())

    mean_probs_mc, std_probs_mc, entropy = mc_dropout_classifier(classifier_model, x_cls, T=40)

    # save volumes
    os.makedirs("results", exist_ok=True)
    vol_dir = os.path.join("results", "volumes")
    save_volume(vol_dir, "J_true_cf", J_true_cf.cpu())
    save_volume(vol_dir, "B_sample_cf", B_sample_cf.cpu())
    save_volume(vol_dir, "J_mean_cf", mean_J.cpu())
    save_volume(vol_dir, "J_std_cf", std_J.cpu())
    save_volume(vol_dir, "J_cred_lo_cf", lo_J.cpu())
    save_volume(vol_dir, "J_cred_hi_cf", hi_J.cpu())

    # build report pdf (multi-page)
    pdf_path = os.path.join("results", "pipeline_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # page 1: overview (single central slice + probs)
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 3)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])

        for ax in [ax0, ax1, ax2, ax3]:
            ax.axis("off")

        mag_true = _vol_to_mag(J_true_cf)
        mag_B = _vol_to_mag(B_sample_cf)
        mag_mean = _vol_to_mag(mean_J)
        mag_std = _vol_to_mag(std_J)

        zc = grid_size // 2
        ax0.imshow(mag_true[:, :, zc])
        ax0.set_title("true |J| (axial slice)")
        ax1.imshow(mag_B[:, :, zc])
        ax1.set_title("noisy |B| (axial slice)")
        ax2.imshow(mag_mean[:, :, zc])
        ax2.set_title("recon mean |J| (axial slice)")
        ax3.imshow(mag_std[:, :, zc])
        ax3.set_title("recon std |J| (axial slice)")

        ax4.bar(diseases, probs.squeeze(0).cpu().numpy())
        ax4.set_ylim([0, 1])
        ax4.set_title(f"class probs (deterministic)\npred={diseases[pred_class]}")

        ax5.axis("off")
        cell_text = [[
            f"{features['mean_magnitude'].item():.4f}",
            f"{features['std_magnitude'].item():.4f}",
            str(features["max_magnitude_coord"].tolist()),
        ]]
        ax5.table(cellText=cell_text, colLabels=["mean |J|", "std |J|", "max coord"], loc="center")
        ax5.set_title("voxel-derived features")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # page 2: multi-slice grids (true / mean / std)
        fig = plot_multislice_grid(mag_true, "true current magnitude |J|", n_slices=5)
        pdf.savefig(fig); plt.close(fig)
        fig = plot_multislice_grid(mag_mean, "reconstructed mean |J| (mc)", n_slices=5)
        pdf.savefig(fig); plt.close(fig)
        fig = plot_multislice_grid(mag_std, "reconstruction uncertainty std(|J|)", n_slices=5)
        pdf.savefig(fig); plt.close(fig)

        # page 3: mips
        fig = plot_mips(mag_true, "true |J|")
        pdf.savefig(fig); plt.close(fig)
        fig = plot_mips(mag_mean, "recon mean |J|")
        pdf.savefig(fig); plt.close(fig)
        fig = plot_mips(mag_std, "recon std |J|")
        pdf.savefig(fig); plt.close(fig)

        # page 4: 3d voxel renders (thresholded)
        fig = plot_voxel_render(mag_true, "3d voxels: true |J|", quantile=0.995)
        pdf.savefig(fig); plt.close(fig)
        fig = plot_voxel_render(mag_mean, "3d voxels: recon mean |J|", quantile=0.995)
        pdf.savefig(fig); plt.close(fig)
        fig = plot_voxel_render(mag_std, "3d voxels: uncertainty std(|J|)", quantile=0.995)
        pdf.savefig(fig); plt.close(fig)

        # page 5: credible interval width (hi - lo) magnitude
        cred_width = _vol_to_mag(hi_J - lo_J)
        fig = plot_multislice_grid(cred_width, "95% credible interval width (|hi-lo|)", n_slices=5)
        pdf.savefig(fig); plt.close(fig)

        # page 6: confusion matrix
        fig = plot_confusion_matrix(cm, diseases, "confusion matrix (val set)")
        pdf.savefig(fig); plt.close(fig)

        # page 7: classifier mc-dropout uncertainty
        fig = plot_prob_uncertainty(mean_probs_mc, std_probs_mc, diseases, f"mc-dropout probs (T=40)\nentropy={entropy:.4f}")
        pdf.savefig(fig); plt.close(fig)

    print(f"saved multi-page report: {pdf_path}")
    print(f"saved volumes to: {vol_dir}")


if __name__ == "__main__":
    main()
