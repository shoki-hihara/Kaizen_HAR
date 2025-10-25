# kaizen/methods/tpn_method.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from kaizen.methods.tpn import TPN


class TPNMethod(pl.LightningModule):
    """
    KaizenのResNetMethodに対応するTPNバックボーン版
    - 特徴抽出器: TPN
    - 分類器: Linear層
    - 出力: {"z": z, "classifier_logits": logits}
      → Kaizen蒸留ラッパが distill_current_key="z", "classifier_logits" にアクセスできる
    """

    def __init__(
        self,
        in_channels=3,
        feature_dim=128,
        num_classes=18,
        lr=1e-3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Backbone (TPN) ----
        self.backbone = TPN(in_channels=in_channels, feature_dim=feature_dim)

        # ---- Classifier ----
        self.classifier = nn.Linear(feature_dim, num_classes)

        # ---- Optimizer設定 ----
        self.lr = lr

    def forward(self, x):
        """
        Kaizen標準の出力フォーマット
        z: 特徴ベクトル
        classifier_logits: クラス分類ロジット
        """
        z = self.backbone(x)
        logits = self.classifier(z)
        return {"z": z, "classifier_logits": logits}

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        logits = outputs["classifier_logits"]
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        logits = outputs["classifier_logits"]
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
