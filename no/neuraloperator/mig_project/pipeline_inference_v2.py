# pipeline_inference_v2.py
#
# full 5-class MIG → inverse FNO → voxel currents → classifier pipeline

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

    # geometry
    Nx, Ny, Nz = 32, 32, 16
    dx = dy = dz = 1e-3
    Nsx, Nsy = 16, 16
    sensor_offset = 0.01

    # scales (MUST match train_inverse_v2 + classifier_v2)
    J_scale = 50.0
    B_scale = 50.0

    # coords
    X, Y, Z = make_volume_coords(Nx, Ny, Nz, dx, dy, dz, device=device)
    sensor_coords, _, _ = make_sensor_plane(
        Nsx, Nsy, X, Y, Z, sensor_offset=sensor_offset, device=device
    )

    print(f"volume grid: {Nx}×{Ny}×{Nz}")
    print(f"sensors: {Nsx}×{Nsy} = {Nsx*Nsy}")

    # inverse model
    inverse_model = MIGInverseModel(
        Nx=Nx, Ny=Ny, Nz=Nz,
        Hs=Nsx, Ws=Nsy,
        latent_channels=24,
        fno_width=48,
        fno_layers=4,
        modes_x=12, modes_y=12, modes_z=8
    ).to(device)
    inverse_state = torch.load("mig_inverse_fno_v2_diseases.pt", map_location=device)
    inverse_model.load_state_dict(inverse_state)
    inverse_model.eval()
    print("✓ loaded inverse model: mig_inverse_fno_v2_diseases.pt")

    # classifier
    classifier = MIGVoxelClassifier(in_channels=3, n_classes=5).to(device)
    clf_state = torch.load("mig_classifier_v2.pt", map_location=device)
    classifier.load_state_dict(clf_state)
    classifier.eval()
    print("✓ loaded classifier: mig_classifier_v2.pt")

    label_names = {
        0: "healthy",
        1: "ischemia",
        2: "arrhythmia",
        3: "block",
        4: "scar",
    }

    n_classes = 5
    N_EVAL = 500
    print(f"\nrunning full 5-class pipeline on {N_EVAL} samples...\n")

    total = 0
    correct = 0
    mse_sum = 0.0
    conf_mat = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    for i in range(N_EVAL):
        # ground-truth currents + label
        J_true, label = generate_disease_sample(Nx, Ny, Nz, device)
        # (3,Nx,Ny,Nz), label in [0..4]

        # forward: J_true -> B
        B = compute_B_field_from_J(J_true, X, Y, Z, sensor_coords)

        # normalize B for inverse model
        Bn = B / B_scale

        # inverse: B_norm -> J_pred_norm -> J_pred
        B_batch = Bn.unsqueeze(0)
        Jn_pred = inverse_model(B_batch, X, Y, Z)[0]    # (3,Nx,Ny,Nz)
        J_pred = Jn_pred * J_scale

        # classifier input
        x = J_pred.unsqueeze(0)  # (1,3,Nx,Ny,Nz)
        logits = classifier(x)
        pred = int(logits.argmax(dim=1).item())

        # reconstruction error
        mse = F.mse_loss(J_pred, J_true).item()

        total += 1
        correct += int(pred == label)
        mse_sum += mse

        conf_mat[label, pred] += 1

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"[{i+1}/{N_EVAL}] "
                f"true={label_names[label]} | pred={label_names[pred]} | mse={mse:.6f}"
            )

    avg_mse = mse_sum / total
    acc = correct / total

    print("\n================= PIPELINE V2 RESULTS =================")
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
