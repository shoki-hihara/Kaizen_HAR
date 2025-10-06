import torch
import torch.nn as nn
import torch.nn.functional as F

class TPN(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, feature_dim=128):
        super().__init__()
        # 時系列入力に対する1D畳み込み層
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        # x: (batch, channel, timesteps)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x
