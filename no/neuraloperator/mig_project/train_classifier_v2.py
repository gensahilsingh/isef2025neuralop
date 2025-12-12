# train_classifier_v3.py
#
# Fully upgraded classifier training on J_pred from inverse model
# 5 classes: healthy, ischemia, arrhythmia, block, scar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from inverse_model import MIGInverseModel
from disease_patterns import generate_disease_sample
from mig_forward import compute_B_field_from_J
from mig_geometry import make_volume_coords, make_sensor_plane
from classifier3d import Classifier3D


# ================================================================
# DATASET — generates disease → forward B → inverse model → J_pred
# ================================================================
class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, Nx, Ny, Nz,
                 dx, dy, dz, Nsx, Nsy, sensor_offset,
                 inverse_model, device):

        self.n_samples = n_samples
        self.device = device

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Nsx, self.Nsy = Nsx, Nsy

        # load geometry
        X, Y, Z = make_volume_coords(Nx, Ny, Nz, dx, dy, dz, device)
        sensor_coords, _, _ = make_sensor_plane(Nsx, Nsy, X, Y, Z,
                                                sensor_offset=sensor_offset,
                                                device=device)

        self.X, self.Y, self.Z = X, Y, Z
        self.sensor_coords = sensor_coords

        self.inverse_model = inverse_model
        self.inverse_model.eval()

        # normalization constants
        self.J_scale = 50.0
        self.B_scale = 50.0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 1. Create disease ground-truth J
        J_true, label = generate_disease_sample(self.Nx, self.Ny, self.Nz, self.device)

        # 2. Forward physics → B field
        B = compute_B_field_from_J(J_true, self.X, self.Y, self.Z, self.sensor_coords)

        # 3. Normalize
        Bn = B / self.B_scale

        # 4. Run inverse model to get J_pred
        with torch.no_grad():
            J_pred = self.inverse_model(Bn.unsqueeze(0), self.X, self.Y, self.Z)[0]

        # classifier input = J_pred (already normalized by inverse model behavior)
        return {
            "J": J_pred,
            "label": label
        }


# ================================================================
# TRAINING LOOP
# ================================================================
def train_classifier_v3():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    # geometry
    Nx, Ny, Nz = 32, 32, 16
    dx = dy = dz = 1e-3
    Nsx, Nsy = 16, 16
    sensor_offset = 0.01

    # load inverse model (frozen)
    inverse_model = MIGInverseModel(
        Nx=Nx, Ny=Ny, Nz=Nz,
        Hs=Nsx, Ws=Nsy,
        latent_channels=24,
        fno_width=48,
        fno_layers=4,
        modes_x=12, modes_y=12, modes_z=8
    ).to(device)

    inverse_model.load_state_dict(torch.load("mig_inverse_fno_v2_diseases.pt"))
    for p in inverse_model.parameters():
        p.requires_grad = False
    print("✓ loaded and froze inverse model")

    # dataset sizes
    train_samples = 8000
    val_samples = 2000
    batch_size = 4

    # datasets
    train_set = ClassifierDataset(
        n_samples=train_samples,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        Nsx=Nsx, Nsy=Nsy,
        sensor_offset=sensor_offset,
        inverse_model=inverse_model,
        device=device
    )

    val_set = ClassifierDataset(
        n_samples=val_samples,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        Nsx=Nsx, Nsy=Nsy,
        sensor_offset=sensor_offset,
        inverse_model=inverse_model,
        device=device
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # classifier model (5 disease classes)
    classifier = Classifier3D(num_classes=5).to(device)

    # class weights (slightly balancing)
    weights = torch.tensor([1.0, 1.1, 1.2, 1.1, 1.0], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-4)

    print("starting classifier_v3 training...")

    EPOCHS = 40

    for epoch in range(1, EPOCHS + 1):
        classifier.train()

        running_loss = 0.0
        correct = 0

        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}"):
            J = batch["J"].to(device)  # (B,3,Nx,Ny,Nz)
            labels = batch["label"].to(device)

            logits = classifier(J)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * J.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        scheduler.step()

        train_loss = running_loss / train_samples
        train_acc = correct / train_samples * 100

        # ----------------------
        # validation
        # ----------------------
        classifier.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for batch in val_loader:
                J = batch["J"].to(device)
                labels = batch["label"].to(device)

                logits = classifier(J)
                loss = loss_fn(logits, labels)

                val_loss += loss.item() * J.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        val_loss /= val_samples
        val_acc = val_correct / val_samples * 100

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%"
        )

    # save final classifier
    torch.save(classifier.state_dict(), "mig_classifier_v3.pt")
    print("\n✓ saved classifier → mig_classifier_v3.pt")


if __name__ == "__main__":
    train_classifier_v3()
