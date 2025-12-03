import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class TPNBackboneOriginal(nn.Module):
    """
    論文「Multi-task Self-Supervised Learning for Human Activity Detection」の
    TPNバックボーン（Conv部分）をできるだけ忠実に再現したクラス。

    - 入力: (B, C, T) or (B, T, C)
    - 出力: (B, 96)  ※最後のConv(96ch)をGlobal Max Poolしたベクトル
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.1):
        super().__init__()

        # Conv1: in_channels -> 32, kernel_size=24
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=24, stride=1, padding=0)
        # Conv2: 32 -> 64, kernel_size=16
        self.conv2 = nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=0)
        # Conv3: 64 -> 96, kernel_size=8
        self.conv3 = nn.Conv1d(64, 96, kernel_size=8, stride=1, padding=0)

        # 論文ではDropout(0.1)のみでBatchNormは使っていない
        self.dropout = nn.Dropout(p=dropout_p)

        # Global Max Pooling (time軸方向)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # 便利のため、出力次元を明示しておく
        self.features_dim = 96

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) or (B, T, C)
        戻り値: (B, 96)
        """

        # 形状 (B, T, C) の場合は (B, C, T) に並び替え
        if x.ndim != 3:
            raise RuntimeError(f"TPNBackboneOriginal expects 3D input, got {x.ndim}D: {tuple(x.shape)}")

        if x.shape[1] < x.shape[2]:
            # 多くの場合 (B, T, C) を想定
            # 例: (B, 400, 3) -> (B, 3, 400)
            # C < T なら (B, T, C) とみなして transpose
            x = x.transpose(1, 2)  # (B, C, T)

        # Conv1
        x = self.conv1(x)           # (B, 32, T1)
        x = F.relu(x)
        x = self.dropout(x)

        # Conv2
        x = self.conv2(x)           # (B, 64, T2)
        x = F.relu(x)
        x = self.dropout(x)

        # Conv3
        x = self.conv3(x)           # (B, 96, T3)
        x = F.relu(x)
        x = self.dropout(x)

        # Global Max Pooling over time dimension
        x = self.global_pool(x)     # (B, 96, 1)
        x = x.squeeze(-1)           # (B, 96)

        return x


class TPNMultiTaskHead(nn.Module):
    """
    TPNの自己教師ありタスク用ヘッド群。
    - 各タスクごとに FC(256) -> ReLU -> FC(1) を持つ。
    - 最終層の出力はlogits (生のスコア) とし、Sigmoidは外でかける想定。
    """

    def __init__(self, in_features: int = 96, hidden_dim: int = 256, num_tasks: int = 8):
        super().__init__()
        self.num_tasks = num_tasks

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, in_features)
        戻り値: logits (B, num_tasks)
        """
        logits_per_task = []
        for head in self.heads:
            logit = head(z)             # (B, 1)
            logits_per_task.append(logit)

        # (B, num_tasks, 1) -> (B, num_tasks)
        logits = torch.cat(logits_per_task, dim=1)
        return logits


class TPNOriginal(nn.Module):
    """
    論文準拠 TPN: バックボーン + マルチタスク自己教師ありヘッド
    """

    def __init__(self, in_channels: int = 3, num_tasks: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.backbone = TPNBackboneOriginal(in_channels=in_channels, dropout_p=dropout_p)
        self.head = TPNMultiTaskHead(in_features=self.backbone.features_dim,
                                     hidden_dim=256,
                                     num_tasks=num_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) or (B, T, C)
        戻り値: logits_selfsup (B, num_tasks)
        """
        z = self.backbone(x)      # (B, 96)
        logits = self.head(z)     # (B, num_tasks)
        return logits

class TPNSSLModule(pl.LightningModule):
    """
    TPN 論文準拠の自己教師あり学習用 LightningModule の例。
    - 入力: (x, y_multi)
      x: (B, C, T) / (B, T, C)
      y_multi: (B, num_tasks)  ... 各タスクのラベル (0/1)
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_tasks: int = 8,
                 lr: float = 3e-4,
                 weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = TPNOriginal(in_channels=in_channels, num_tasks=num_tasks)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")  # 後でタスクごと平均するため

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # y: (B, num_tasks)  (0/1)
        logits = self(x)  # (B, num_tasks)

        # BCEWithLogitsLoss: (B, num_tasks)
        loss_per_entry = self.criterion(logits, y.float())
        loss = loss_per_entry.mean()  # バッチとタスク方向で平均

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        return optimizer


class TPNActivityNetOriginal(nn.Module):
    """
    TPN 論文の Activity Recognition ネットワーク（下流）。
    - Conv部は TPNBackboneOriginal と同じ構造。
    - 研究では事前学習済みTPNのConv重みを転移して利用。
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 18,
                 dropout_p: float = 0.1,
                 use_pretrained_backbone: bool = False,
                 pretrained_backbone: TPNBackboneOriginal = None):
        super().__init__()

        if use_pretrained_backbone and pretrained_backbone is not None:
            # 事前学習済みバックボーンをそのまま使う
            self.backbone = pretrained_backbone
        else:
            # ゼロから学習する場合
            self.backbone = TPNBackboneOriginal(in_channels=in_channels, dropout_p=dropout_p)

        # 論文では 1024ユニットのFCを通してからクラス分類
        self.fc1 = nn.Linear(self.backbone.features_dim, 1024)
        self.fc_out = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) or (B, T, C)
        戻り値: logits (B, num_classes)
        """
        z = self.backbone(x)          # (B, 96)
        z = F.relu(self.fc1(z))       # (B, 1024)
        logits = self.fc_out(z)       # (B, num_classes)
        return logits
