# train_classifier.py

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from mig_dataset import MIGClassifierDataset
from classifier3d import VoxelDiseaseClassifier3D


def train_classifier():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    Nx, Ny, Nz = 32, 32, 16
    dx = dy = dz = 1e-3

    n_train = 2000
    n_val = 400
    batch_size = 4

    train_set = MIGClassifierDataset(
        n_samples=n_train,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        device=device
    )
    val_set = MIGClassifierDataset(
        n_samples=n_val,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz,
        device=device
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = VoxelDiseaseClassifier3D(in_channels=3, num_classes=4, base_channels=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    n_epochs = 30

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        n_train_samples = 0

        for batch in train_loader:
            J = batch["J"].to(device)            # (b,3,Nx,Ny,Nz)
            labels = batch["label"].to(device)   # (b,)

            optimizer.zero_grad()

            logits = model(J)                    # (b,num_classes)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * J.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            n_train_samples += J.size(0)

        train_loss = train_loss_sum / n_train_samples
        train_acc = train_correct / n_train_samples

        # validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        n_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                J = batch["J"].to(device)
                labels = batch["label"].to(device)
                logits = model(J)
                loss = F.cross_entropy(logits, labels)

                val_loss_sum += loss.item() * J.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                n_val_samples += J.size(0)

        val_loss = val_loss_sum / n_val_samples
        val_acc = val_correct / n_val_samples

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.5f}, val_acc={val_acc:.3f}"
        )

    torch.save(model.state_dict(), "mig_voxel_classifier.pt")
    print("saved classifier to mig_voxel_classifier.pt")


if __name__ == "__main__":
    train_classifier()
