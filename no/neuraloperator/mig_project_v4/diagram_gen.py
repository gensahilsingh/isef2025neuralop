import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

# assumes you run this inside your project folder (same place train.py is)
from synthetic_data import generate_dataset
from biot_savart import compute_biot_savart


def mag(vol_xyz3: torch.Tensor) -> np.ndarray:
    """vol_xyz3: (X,Y,Z,3) -> numpy (X,Y,Z) magnitude"""
    m = torch.sqrt((vol_xyz3**2).sum(dim=-1) + 1e-12)
    return m.detach().cpu().numpy()


def make_box1_figure(
    out_path="fig_box1_synthetic_sim.png",
    grid_size=32,
    disease="ischemia",
    dx=1.0,
    noise_level=0.0,
    sensor_planes=("top",),
    sensor_gap=1.5,
    sensor_dropout=0.0,
    seed=7,
):
    # --- generate one sample of J and two kinds of observations ---
    diseases = ["normal", "ischemia", "arrhythmia", "hypertrophy"]
    assert disease in diseases

    # (a) dense voxel observation (for showing the actual forward field)
    currents_voxel, fields_voxel, labels = generate_dataset(
        dataset_size=1,
        grid_size=grid_size,
        diseases=diseases,
        noise_level=noise_level,
        dx=dx,
        seed=seed,
        observation="voxel",          # dense B volume
        device="cpu",
    )

    # (b) sensor-plane embedded observation (what your model actually sees)
    currents_sens, fields_sens, _ = generate_dataset(
        dataset_size=1,
        grid_size=grid_size,
        diseases=diseases,
        noise_level=noise_level,
        dx=dx,
        seed=seed,                    # keep same seed so the underlying sample matches
        observation="sensors",         # sparse embedded planes
        sensor_planes=sensor_planes,
        sensor_gap=sensor_gap,
        sensor_dropout=sensor_dropout,
        device="cpu",
    )

    # pick the disease you want by regenerating until label matches (clean + deterministic)
    # (your generate_dataset chooses diseases randomly; this forces a specific one)
    target = diseases.index(disease)
    tries = 0
    while labels[0] != target and tries < 200:
        seed += 1
        currents_voxel, fields_voxel, labels = generate_dataset(
            dataset_size=1, grid_size=grid_size, diseases=diseases,
            noise_level=noise_level, dx=dx, seed=seed, observation="voxel", device="cpu"
        )
        currents_sens, fields_sens, _ = generate_dataset(
            dataset_size=1, grid_size=grid_size, diseases=diseases,
            noise_level=noise_level, dx=dx, seed=seed, observation="sensors",
            sensor_planes=sensor_planes, sensor_gap=sensor_gap, sensor_dropout=sensor_dropout,
            device="cpu"
        )
        tries += 1

    J = currents_voxel[0]        # (X,Y,Z,3)
    B_dense = fields_voxel[0]    # (X,Y,Z,3) dense B
    B_sensors = fields_sens[0]   # (X,Y,Z,3) sparse embedded plane(s)

    # sanity check: compute B from J too (optional) — useful if you want to show “physics forward”
    B_from_J = compute_biot_savart(J.unsqueeze(0)).squeeze(0)  # (X,Y,Z,3)

    # --- prepare slices ---
    X, Y, Z, _ = J.shape
    zc = Z // 2

    Jmag = mag(J)
    Bmag = mag(B_dense)
    Bmag_phys = mag(B_from_J)
    Bsens_mag = mag(B_sensors)

    # quiver arrows (downsample so it looks clean)
    step = max(1, grid_size // 16)  # ~16 arrows across
    xs = np.arange(0, X, step)
    ys = np.arange(0, Y, step)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    Jx = J[..., 0].detach().cpu().numpy()
    Jy = J[..., 1].detach().cpu().numpy()
    U = Jx[XX, YY, zc]
    V = Jy[XX, YY, zc]

    # sensor mip for “sparsity pop”
    mip_sensors_xy = Bsens_mag.max(axis=2)

    # --- build figure (poster-friendly) ---
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # (1) |J| + quiver
    im1 = ax1.imshow(Jmag[:, :, zc], origin="lower")
    ax1.quiver(YY, XX, V, U, angles="xy", scale_units="xy", scale=None, width=0.0025)
    ax1.set_title(f"synthetic intracellular current |J| (axial z={zc}) + vectors")
    ax1.set_xticks([]); ax1.set_yticks([])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    # (2) dense |B| from generator
    im2 = ax2.imshow(Bmag[:, :, zc], origin="lower")
    ax2.set_title("dense magnetic field |B| (generator forward)")
    ax2.set_xticks([]); ax2.set_yticks([])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    # (3) dense |B| from biot–savart(J) (physics consistency visual)
    im3 = ax3.imshow(Bmag_phys[:, :, zc], origin="lower")
    ax3.set_title("dense |B| computed from J via biot–savart")
    ax3.set_xticks([]); ax3.set_yticks([])
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)

    # (4) sensor embedded slice (shows plane)
    im4 = ax4.imshow(Bsens_mag[:, :, zc], origin="lower")
    ax4.set_title(f"sensor observation |B| (embedded) slice z={zc}")
    ax4.set_xticks([]); ax4.set_yticks([])
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.02)

    # (5) sensor mip (strong “sparse sensing” look)
    im5 = ax5.imshow(mip_sensors_xy, origin="lower")
    ax5.set_title("sensor observation mip (max over z)")
    ax5.set_xticks([]); ax5.set_yticks([])
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.02)

    # (6) tiny legend / parameters (no heavy text, poster-friendly)
    ax6.axis("off")
    text = (
        f"disease: {disease}\n"
        f"grid: {grid_size}³, dx={dx}\n"
        f"obs: sensor_planes={sensor_planes}\n"
        f"sensor_gap={sensor_gap}, dropout={sensor_dropout}\n"
        f"noise_level={noise_level}\n"
        f"seed={seed}\n"
    )
    ax6.text(0.02, 0.98, text, va="top")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    make_box1_figure(
        out_path="fig_box1_synthetic_sim.png",
        grid_size=32,           # 32 looks way better on a poster than 16
        disease="ischemia",     # change to normal/arrhythmia/hypertrophy
        sensor_planes=("top",), # or ("top","bottom") if you want to show multi-plane sensing
        sensor_gap=1.5,
        sensor_dropout=0.0,
        noise_level=0.0,
        seed=7,
    )
