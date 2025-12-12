import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier3D(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),   # 32→16, 32→16, 16→8

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),   # 16→8, 16→8, 8→4
        )

        self.fc_in_dim = None
        self.classifier_head = None
        self.num_classes = num_classes

    def build_head(self, x):
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        self.fc_in_dim = x_flat.size(1)

        self.classifier_head = nn.Sequential(
            nn.Linear(self.fc_in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, self.num_classes)
        ).to(x.device)

    def forward(self, x):
        x = self.features(x)

        if self.classifier_head is None:
            self.build_head(x)

        B = x.size(0)
        x = x.reshape(B, -1)  # reshape avoids non-contiguous errors

        return self.classifier_head(x)
