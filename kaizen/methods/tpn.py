import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class TPN(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class TPNLightning(pl.LightningModule):
    def __init__(self, in_channels=3, feature_dim=128, num_classes=6, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = TPN(in_channels=in_channels, feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.lr = lr

    def forward(self, x):
        z = self.backbone(x)
        return self.classifier(z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
