# classifier3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelDiseaseClassifier3D(nn.Module):
    """
    3D CNN classifier:
      Input: J voxel field (3,Nx,Ny,Nz)
      Output: logits over disease classes.
    """

    def __init__(self, in_channels=3, num_classes=4, base_channels=16):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.bn2 = nn.BatchNorm3d(base_channels * 2)
        self.bn3 = nn.BatchNorm3d(base_channels * 4)

        self.fc1 = nn.Linear(base_channels * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, J):
        """
        J: (batch,3,Nx,Ny,Nz)
        """
        x = F.gelu(self.bn1(self.conv1(J)))
        x = self.pool(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # shape: (b,C,D,H,W)

        # global average pooling
        x = x.mean(dim=[2, 3, 4])  # (b,C)

        x = F.gelu(self.fc1(x))
        logits = self.fc2(x)
        return logits
