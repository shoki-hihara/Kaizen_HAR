from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple
import functools
import operator
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from kaizen.utils.lars import LARSWrapper
from kaizen.utils.metrics import accuracy_at_k, weighted_mean
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)


class LinearTPNModel(pl.LightningModule):
    def __init__(self, backbone: nn.Module, num_classes: int, **kwargs):
        """TPN用のLinear評価モデル

        Args:
            backbone (nn.Module): TPN backbone
            num_classes (int): クラス数
            kwargs: CLIから渡されるハイパーパラメータ
        """
        super().__init__()

        self.backbone = backbone
        with torch.no_grad():
            # WISDM2019: batch=1, channels=3, seq_len=384
            dummy = torch.randn(1, 3, 384)
            output_dim = backbone(dummy).shape[1]  # チャンネル方向の次元を取得
        self.classifier = nn.Linear(output_dim, num_classes)

        # freeze_backbone を kwargs から取得
        freeze_backbone = kwargs.get("freeze_backbone", False)
        if not freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = True

        # CLI引数をまとめて保持
        self.hparams.update(kwargs)

        self.domains = ["real", "quickdraw", "painting", "sketch", "infograph", "clipart"]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """TPN用にCLI引数を整理"""
        parser = parent_parser.add_argument_group("linear_tpn")

        # optimizer / scheduler
        parser.add_argument("--optimizer", choices=["sgd", "adam"], type=str, required=True)
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--exclude_bias_n_norm", action="store_true")
        parser.add_argument("--scheduler", choices=[
            "reduce", "cosine", "warmup_cosine", "step", "exponential", "none"
        ], default="reduce")
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")

        return parent_parser

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """TPN特徴抽出 + Linear層分類"""
        with torch.no_grad():
            # Conv1dの入力に合わせて次元を転置
            if X.ndim == 3 and X.shape[1] == 384 and X.shape[2] == 3:
                X = X.transpose(1, 2)  # [batch, seq_len, channels] -> [batch, channels, seq_len]
            feats = self.backbone(X)
        logits = self.classifier(feats)
        return {"feats": feats, "logits": logits}

    def configure_optimizers(self):
        """Linear層のみのOptimizer/Scheduler設定"""
        optimizer_name = self.hparams.get("optimizer", "sgd")
        lr = self.hparams.get("lr", 0.1)
        weight_decay = self.hparams.get("weight_decay", 1e-4)
        extra_optimizer_args = self.hparams

        if optimizer_name == "sgd":
            optimizer_cls = torch.optim.SGD
        elif optimizer_name == "adam":
            optimizer_cls = torch.optim.Adam
        else:
            raise ValueError(f"{optimizer_name} not in (sgd, adam)")

        optimizer = optimizer_cls(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **extra_optimizer_args,
        )

        if self.hparams.get("lars", False):
            optimizer = LARSWrapper(optimizer, exclude_bias_n_norm=self.hparams.get("exclude_bias_n_norm", False))

        scheduler_name = self.hparams.get("scheduler", "reduce")
        max_epochs = self.hparams.get("max_epochs", 100)
        lr_decay_steps = self.hparams.get("lr_decay_steps", None)

        if scheduler_name == "none":
            return optimizer
        elif scheduler_name == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, 10, max_epochs)
        elif scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, max_epochs)
        elif scheduler_name == "reduce":
                scheduler = {
                                "scheduler": ReduceLROnPlateau(optimizer, mode="min"),
                                "monitor": "val_loss", 
                                "interval": "epoch",
                                "frequency": 1
                            }
        elif scheduler_name == "step":
            scheduler = MultiStepLR(optimizer, lr_decay_steps, gamma=0.1)
        elif scheduler_name == "exponential":
            scheduler = ExponentialLR(optimizer, weight_decay)
        else:
            raise ValueError(f"{scheduler_name} not supported")

        return [optimizer], [scheduler]

    def shared_step(self, batch: Tuple, batch_idx: int):
        *_, X, target = batch
        batch_size = X.size(0)

        logits = self(X)["logits"]
        loss = F.cross_entropy(logits, target)
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))
        return batch_size, loss, acc1, acc5, logits

    def training_step(self, batch, batch_idx):
        self.backbone.eval()
        _, loss, acc1, acc5, _ = self.shared_step(batch, batch_idx)
        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size, loss, acc1, acc5, logits = self.shared_step(batch, batch_idx)
        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "logits": logits,
            "targets": batch[-1],
        }
        if self.hparams.get("split_strategy", "class") == "domain" and len(batch) == 3:
            results["domains"] = batch[0]
        return results

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")
        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        if not self.trainer.sanity_checking:
            preds = torch.cat([o["logits"].max(-1)[1] for o in outs]).cpu().numpy()
            targets = torch.cat([o["targets"] for o in outs]).cpu().numpy()
            mask_correct = preds == targets

            split_strategy = self.hparams.get("split_strategy", "class")
            tasks = self.hparams.get("tasks", None)

            if split_strategy == "class" and tasks is not None:
                for task_idx, task in enumerate(tasks):
                    mask_task = np.isin(targets, np.array(task))
                    correct_task = np.logical_and(mask_task, mask_correct).sum()
                    log[f"val_acc1_task{task_idx}"] = correct_task / mask_task.sum()

            if split_strategy == "domain":
                domains_list = [o["domains"] for o in outs]
                domains_list = np.array(functools.reduce(operator.iconcat, domains_list, []))
                for task_idx, domain in enumerate(self.domains):
                    mask_domain = np.isin(domains_list, np.array([domain]))
                    correct_domain = np.logical_and(mask_domain, mask_correct).sum()
                    log[f"val_acc1_{domain}_{task_idx}"] = correct_domain / mask_domain.sum()

        self.log_dict(log, sync_dist=True)
