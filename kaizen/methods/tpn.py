# kaizen/methods/tpn_method.py
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
        """
        想定される入力パターン：
        - (B, 384, 3): WISDM の元データ [seq_len, channels]
        - (B, 3, 384): すでに Conv1d 用に並び替え済み
        - (B, 1, 384, 3): frozen_encoder 経由で入ってきている現在のケース
        """

        # --- 4次元 [B, 1, 384, 3] への対応 ---
        if x.ndim == 4:
            # 期待通り「ダミーチャンネル 1」が立っている場合
            if x.shape[1] == 1 and x.shape[2] == 384 and x.shape[3] == 3:
                # (B, 1, 384, 3) → (B, 384, 3)
                x = x.squeeze(1)
            else:
                # 想定外の形が来た場合は、デバッグがしやすいように明示的にエラー
                raise RuntimeError(f"TPN got 4D input with unexpected shape: {tuple(x.shape)}")

        # --- ここから 3次元の処理 ---
        if x.ndim != 3:
            raise RuntimeError(f"TPN expects 3D input after preprocessing, but got {x.ndim}D: {tuple(x.shape)}")

        # x: (B, 384, 3) → (B, 3, 384) に並び替え
        if x.shape[1] == 384 and x.shape[2] == 3:
            x = x.permute(0, 2, 1)  # (B, 3, 384)
        # すでに (B, 3, 384) ならそのまま
        elif x.shape[1] == 3 and x.shape[2] == 384:
            pass
        else:
            raise RuntimeError(f"TPN got 3D input with unexpected shape: {tuple(x.shape)}")

        # Conv1d パス
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)           # (B, 128, 1)
        x = x.squeeze(-1)          # (B, 128)
        x = self.fc(x)             # (B, feature_dim)
        return x

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
