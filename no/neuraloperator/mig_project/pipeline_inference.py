# pipeline_inference.py
#
# full MIG → inverse FNO → voxel currents → classifier pipeline
# evaluates reconstruction + disease classification performance

import torch
import torch.nn.functional as F

from inverse_model import MIGInverseModel
from classifier3d import MIGVoxelClassifier
from mig_geometry import make_volume_coords, make_sensor_plane
from mig_forward import compute_B_field_from_J
from disease_patterns import generate_disease_sample


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    # -------------------------------------------------------------
    # geometry / constants (must match training settings)
    # -------------------------------------------------------------
    Nx, Ny, Nz = 32, 32, 16
    dx = dy = dz = 1e-3

    Nsx, Nsy = 16, 16
    sensor_offset = 0.01

    # -------------------------------------------------------------
    # coords and sensor layout
    # -------------------------------------------------------------
    X, Y, Z = make_volume_coords(Nx, Ny, Nz, dx, dy, dz, device=device)
    sensor_coords, Hs, Ws = make_sensor_plane(
        Nsx, Nsy, X, Y, Z, sensor_offset=sensor_offset, device=device
    )

    print(f"volume grid: {Nx}×{Ny}×{Nz}")
    print(f"sensors: {Nsx}×{Nsy} = {Nsx*Nsy}")

    # -------------------------------------------------------------
    # load inverse model (pretrained FNO)
    # -------------------------------------------------------------
    inverse_model = MIGInverseModel(
        Nx=Nx, Ny=Ny, Nz=Nz,
        Hs=Nsx, Ws=Nsy,
        latent_channels=20,
        fno_width=36,
        fno_layers=4,
        modes_x=12, modes_y=12, modes_z=8,
    ).to(device)

    inverse_state = torch.load("mig_inverse_fno_v2.pt", map_location=device)
    inverse_model.load_state_dict(inverse_state)
    inverse_model.eval()
    print("✓ loaded inverse model: mig_inverse_fno_v2.pt")

    # -------------------------------------------------------------
    # load classifier
    # -------------------------------------------------------------
    classifier = MIGVoxelClassifier(in_channels=6, n_classes=5).to(device)
    clf_state = torch.load("mig_classifier_final.pt", map_location=device)
    classifier.load_state_dict(clf_state)
    classifier.eval()
    print("✓ loaded classifier: mig_classifier_final.pt")

    n_classes = classifier.fc.out_features if hasattr(classifier, "fc") else 5

    # label → name mapping (adjust if you changed classes)
    label_names = {
        0: "healthy",
        1: "ischemia",
        2: "arrhythmia",
        3: "class_3_unused",
        4: "class_4_unused",
    }

    # -------------------------------------------------------------
    # evaluation loop
    # -------------------------------------------------------------
    N_EVAL = 500  # number of synthetic test samples
    print(f"\nrunning full pipeline on {N_EVAL} samples...\n")

    total = 0
    correct = 0
    mse_sum = 0.0

    # confusion matrix: true (rows) vs predicted (cols)
    conf_mat = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    for i in range(N_EVAL):
        # 1) generate synthetic disease currents + label
        J_true, label = generate_disease_sample(Nx, Ny, Nz, device)
        # J_true: (3,Nx,Ny,Nz), label: int 0..2

        # 2) forward model: currents → magnetic field at sensors
        B = compute_B_field_from_J(J_true, X, Y, Z, sensor_coords)  # (3,Hs,Ws)

        # 3) run inverse model: B → reconstructed currents J_pred
        B_batch = B.unsqueeze(0)  # (1,3,Hs,Ws)
        J_pred_batch = inverse_model(B_batch, X, Y, Z)  # (1,3,Nx,Ny,Nz)
        J_pred = J_pred_batch[0]  # (3,Nx,Ny,Nz)

        # 4) build classifier input: concat [J_pred, J_true] along channel dim
        x = torch.cat([J_pred, J_true], dim=0).unsqueeze(0)  # (1,6,Nx,Ny,Nz)

        # 5) classifier prediction
        logits = classifier(x)
        pred = int(logits.argmax(dim=1).item())

        # 6) reconstruction error
        mse = F.mse_loss(J_pred, J_true).item()

        # 7) stats
        total += 1
        correct += int(pred == label)
        mse_sum += mse

        if label < n_classes and pred < n_classes:
            conf_mat[label, pred] += 1

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"[{i+1}/{N_EVAL}] true={label_names.get(label, label)} "
                f"pred={label_names.get(pred, pred)} | mse={mse:.5f}"
            )

    # -------------------------------------------------------------
    # final stats
    # -------------------------------------------------------------
    avg_mse = mse_sum / total
    acc = correct / total

    print("\n================= PIPELINE RESULTS =================")
    print(f"Total samples     : {total}")
    print(f"Overall accuracy  : {acc*100:.2f}%")
    print(f"Mean MSE (J_pred) : {avg_mse:.6f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(conf_mat.cpu().numpy())

    print("\nclass indices:")
    for k, v in label_names.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
