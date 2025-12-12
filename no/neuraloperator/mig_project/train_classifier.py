import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from inverse_model import MIGInverseModel
from disease_dataset import DiseaseDataset
from mig_geometry import make_volume_coords
from classifier3d import MIGVoxelClassifier


def train_classifier():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    # geometry
    Nx, Ny, Nz = 32, 32, 16
    dx = dy = dz = 1e-3
    Nsx, Nsy = 16, 16
    sensor_offset = 0.01

    # dataset sizes
    n_train = 4000
    n_val = 1000
    batch_size = 4

    # build coord grid once (for inverse model)
    X, Y, Z = make_volume_coords(Nx, Ny, Nz, dx, dy, dz, device=device)

    # load trained inverse model
    inverse_model = MIGInverseModel(
        Nx=Nx, Ny=Ny, Nz=Nz,
        Hs=Nsx, Ws=Nsy,
        latent_channels=20,
        fno_width=36,
        fno_layers=4,
        modes_x=12, modes_y=12, modes_z=8,
    ).to(device)
    inverse_model.load_state_dict(torch.load("mig_inverse_fno_v2.pt", map_location=device))
    inverse_model.eval()
    print("✓ loaded inverse model")

    # disease datasets (B, J_true, label)
    train_set = DiseaseDataset(
    n_samples=n_train,
    Nx=Nx, Ny=Ny, Nz=Nz,
    dx=dx, dy=dy, dz=dz,
    Nsx=Nsx, Nsy=Nsy,
    sensor_offset=sensor_offset,
    device=device
    )

    val_set = DiseaseDataset(
        n_samples=n_val,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        Nsx=Nsx, Nsy=Nsy,
        sensor_offset=sensor_offset,
        device=device
    )


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # classifier: takes 6 channels (J_pred + J_true), outputs 5 classes
    classifier = MIGVoxelClassifier(in_channels=6, n_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-5)
    n_epochs = 60

    print("starting classifier training...")

    for epoch in range(1, n_epochs + 1):
        # ---- TRAIN ----
        classifier.train()
        train_loss_sum = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{n_epochs}"):
            B = batch["B"].to(device)             # (B,3,Hs,Ws)
            J_true = batch["J"].to(device)   # (B,3,Nx,Ny,Nz)
            labels = batch["label"].to(device)    # (B,)

            optimizer.zero_grad()

            # inverse model: B -> J_pred (no grad)
            with torch.no_grad():
                J_pred = inverse_model(B, X, Y, Z)   # (B,3,Nx,Ny,Nz)

            # concat [J_pred, J_true] along channel axis
            x = torch.cat([J_pred, J_true], dim=1)   # (B,6,Nx,Ny,Nz)

            logits = classifier(x)                   # (B,5)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * B.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = train_loss_sum / total
        train_acc = correct / total

        # ---- VALIDATE ----
        classifier.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                B = batch["B"].to(device)
                J_true = batch["J"].to(device)
                labels = batch["label"].to(device)

                J_pred = inverse_model(B, X, Y, Z)
                x = torch.cat([J_pred, J_true], dim=1)

                logits = classifier(x)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * B.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        # optional checkpointing
        if epoch % 20 == 0:
            torch.save(classifier.state_dict(), f"mig_classifier_ep{epoch}.pt")
            print(f"saved classifier checkpoint: mig_classifier_ep{epoch}.pt")

    torch.save(classifier.state_dict(), "mig_classifier_final.pt")
    print("saved final classifier -> mig_classifier_final.pt")


if __name__ == "__main__":
    train_classifier()
