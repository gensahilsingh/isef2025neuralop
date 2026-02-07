import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# your project's biot_savart.py

# optional: use your synthetic generator if available
try:
    from synthetic_data import generate_dataset
    _HAS_SYNTH = True
except Exception:
    _HAS_SYNTH = False


def mag_cf(vol_cf: torch.Tensor) -> np.ndarray:
    """
    vol_cf: (3, X, Y, Z) cpu/torch
    returns magnitude (X, Y, Z) numpy
    """
    m = torch.sqrt((vol_cf ** 2).sum(dim=0) + 1e-12)
    return m.detach().cpu().numpy()


def load_J_from_pt(j_path: str) -> torch.Tensor:
    """
    loads a saved tensor that is either:
      (3,X,Y,Z) or (X,Y,Z,3)
    returns channel-first (3,X,Y,Z) on cpu
    """
    J = torch.load(j_path, map_location="cpu")
    if isinstance(J, np.ndarray):
        J = torch.from_numpy(J)
    if J.dim() == 4 and J.shape[0] == 3:
        return J.float().contiguous()
    if J.dim() == 4 and J.shape[-1] == 3:
        return J.permute(3, 0, 1, 2).float().contiguous()
    raise ValueError(f"unexpected J shape in {j_path}: {tuple(J.shape)}")


def pick_one_sample_of_disease(
    grid_size: int,
    disease: str,
    seed: int,
    noise_level: float,
):
    """
    uses your synthetic_data.generate_dataset if it exists.
    returns J_cf (3,X,Y,Z) cpu.
    """
    if not _HAS_SYNTH:
        raise RuntimeError("synthetic_data.generate_dataset not importable; use J_path instead.")

    diseases = ["normal", "ischemia", "arrhythmia", "hypertrophy"]
    assert disease in diseases, f"disease must be one of {diseases}"

    target = diseases.index(disease)
    tries = 0
    s = seed

    # generate until we get the right disease label (since your generator picks randomly)
    while True:
        currents, fields, labels = generate_dataset(
            dataset_size=1,
            grid_size=grid_size,
            diseases=diseases,
            noise_level=noise_level,
            seed=s,
        )
        # currents[0] is commonly (X,Y,Z,3) in your pipeline
        y = int(labels[0])
        if y == target:
            J_xyz3 = currents[0].float().contiguous()
            if J_xyz3.shape[-1] != 3:
                raise ValueError(f"expected currents[0] to be (X,Y,Z,3), got {tuple(J_xyz3.shape)}")
            J_cf = J_xyz3.permute(3, 0, 1, 2).contiguous()  # (3,X,Y,Z)
            return J_cf, s

        tries += 1
        s += 1
        if tries > 300:
            raise RuntimeError("could not sample requested disease after many tries; check generator.")


def make_box1_figure(
    out_path="fig_box1_synthetic_sim.png",
    grid_size=32,
    disease="ischemia",
    seed=7,

    # if you already have a real J volume saved from your pipeline, pass it here:
    J_path: str | None = None,

    # measurement realism knobs
    add_noise=True,
    noise_level=0.03,        # fraction of signal std used in add_measurement_noise
    correlated_noise=True,
    corr_scale=0.3,

    # sensor realism knobs
    n_sensors=800,           # total sensor points (sparse)
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ---- get J (channel-first) ----
    if J_path is not None:
        J_cf = load_J_from_pt(J_path)
        if J_cf.shape[1:] != (grid_size, grid_size, grid_size):
            # auto-fix grid_size if user passed mismatched size
            grid_size = int(J_cf.shape[1])
    else:
        J_cf, seed_used = pick_one_sample_of_disease(
            grid_size=grid_size,
            disease=disease,
            seed=seed,
            noise_level=0.0,  # keep J itself clean; we add noise to B below
        )
        seed = seed_used

    # ---- forward biot–savart: J -> B (dense) ----
    op = BiotSavartOperator(grid_size=grid_size, device="cpu")
    B_dense_cf = op(J_cf)  # since J_cf is (3,X,Y,Z), operator returns (3,X,Y,Z)

    # ---- add measurement noise (optional) ----
    if add_noise:
        B_noisy_cf = add_measurement_noise(
            B_dense_cf,
            noise_level=float(noise_level),
            correlated=bool(correlated_noise),
            corr_scale=float(corr_scale),
        )
    else:
        B_noisy_cf = B_dense_cf.clone()

    # ---- apply sparse sensor mask ----
    mask, indices = create_sensor_mask(grid_size=grid_size, n_sensors=int(n_sensors), device="cpu")  # mask: (1,X,Y,Z)
    B_sensors_cf = apply_sensor_mask(B_noisy_cf, mask)  # broadcasting mask onto (3,X,Y,Z)

    # ---- prep visuals ----
    Jmag = mag_cf(J_cf)
    Bmag_dense = mag_cf(B_dense_cf)
    Bmag_noisy = mag_cf(B_noisy_cf)
    Bmag_sensors = mag_cf(B_sensors_cf)

    zc = grid_size // 2

    # quiver on J (only x,y components, downsampled)
    step = max(1, grid_size // 16)  # ~16 arrows across
    xs = np.arange(0, grid_size, step)
    ys = np.arange(0, grid_size, step)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    Jx = J_cf[0].numpy()
    Jy = J_cf[1].numpy()
    U = Jx[XX, YY, zc]
    V = Jy[XX, YY, zc]

    # sensor MIP to scream "sparse"
    mip_sensors_xy = Bmag_sensors.max(axis=2)

    # ---- figure layout ----
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # (1) |J| + vectors
    im1 = ax1.imshow(Jmag[:, :, zc], origin="lower")
    ax1.quiver(YY, XX, V, U, angles="xy", scale_units="xy", scale=None, width=0.0025)
    ax1.set_title(f"synthetic intracellular current |J| (axial z={zc}) + vectors")
    ax1.set_xticks([]); ax1.set_yticks([])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    # (2) dense |B| (clean)
    im2 = ax2.imshow(Bmag_dense[:, :, zc], origin="lower")
    ax2.set_title("biot–savart forward: dense |B| (clean)")
    ax2.set_xticks([]); ax2.set_yticks([])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    # (3) dense |B| (noisy)
    im3 = ax3.imshow(Bmag_noisy[:, :, zc], origin="lower")
    ax3.set_title("dense |B| + measurement noise")
    ax3.set_xticks([]); ax3.set_yticks([])
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)

    # (4) sparse sensor observation slice
    im4 = ax4.imshow(Bmag_sensors[:, :, zc], origin="lower")
    ax4.set_title(f"sparse sensor observation |B| (slice z={zc})")
    ax4.set_xticks([]); ax4.set_yticks([])
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.02)

    # (5) sparse sensor MIP
    im5 = ax5.imshow(mip_sensors_xy, origin="lower")
    ax5.set_title("sensor observation MIP (max over z)")
    ax5.set_xticks([]); ax5.set_yticks([])
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.02)

    # (6) small parameter box (minimal words, still “real”)
    ax6.axis("off")
    text = (
        f"disease: {disease}\n"
        f"grid: {grid_size}³\n"
        f"seed: {seed}\n"
        f"biot–savart: fft operator\n"
        f"noise: {'on' if add_noise else 'off'} "
        f"(level={noise_level}, corr={correlated_noise}, scale={corr_scale})\n"
        f"sensors: n={n_sensors} points\n"
    )
    ax6.text(0.02, 0.98, text, va="top")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    # option a: generate from your synthetic generator
    make_box1_figure(
        out_path="fig_box1_synthetic_sim.png",
        grid_size=32,
        disease="ischemia",
        seed=7,
        J_path=None,
        add_noise=True,
        noise_level=0.03,
        correlated_noise=True,
        corr_scale=0.3,
        n_sensors=800,
    )

    # option b: use your pipeline-saved current (uncomment + set path)
    # make_box1_figure(
    #     out_path="fig_box1_from_saved_J.png",
    #     J_path="results/volumes/J_true_cf.pt",
    #     grid_size=32,      # will auto-fix if mismatched
    #     disease="(from saved)",
    #     add_noise=True,
    #     noise_level=0.03,
    #     n_sensors=800,
    # )
