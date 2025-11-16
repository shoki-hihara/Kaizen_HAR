from argparse import ArgumentParser
from torch.utils.data import DataLoader
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
    def __init__(self, backbone: nn.Module, num_classes: int, past_task_loaders: Optional[List[DataLoader]] = None, **kwargs):
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
        self.past_task_loaders = past_task_loaders

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
        # ★no_grad()を削除
        if X.ndim == 3 and X.shape[1] == 384 and X.shape[2] == 3:
            X = X.transpose(1, 2)
        feats = self.backbone(X)
        logits = self.classifier(feats)
        return {"feats": feats, "logits": logits}

    def configure_optimizers(self):
        optimizer_name = self.hparams.get("optimizer", "sgd")
        lr = self.hparams.get("lr", 0.1)
        weight_decay = self.hparams.get("weight_decay", 1e-4)
    
        if optimizer_name == "sgd":
            optimizer_cls = torch.optim.SGD
            optimizer = optimizer_cls(
                self.classifier.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=self.hparams.get("momentum", 0.9),
                nesterov=self.hparams.get("nesterov", False),
            )
        elif optimizer_name == "adam":
            optimizer_cls = torch.optim.Adam
            optimizer = optimizer_cls(
                self.classifier.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.hparams.get("betas", (0.9, 0.999)),
            )
        else:
            raise ValueError(f"{optimizer_name} not in (sgd, adam)")
    
        # --- LARS wrapper（必要なら）---
        if self.hparams.get("lars", False):
            optimizer = LARSWrapper(
                optimizer,
                exclude_bias_n_norm=self.hparams.get("exclude_bias_n_norm", False),
            )
    
        # --- Scheduler設定 ---
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
                "frequency": 1,
            }
        elif scheduler_name == "step":
            scheduler = MultiStepLR(optimizer, lr_decay_steps, gamma=0.1)
        elif scheduler_name == "exponential":
            scheduler = ExponentialLR(optimizer, 0.95)
        else:
            raise ValueError(f"{scheduler_name} not supported")
    
        return [optimizer], [scheduler]



    def shared_step(self, batch: Tuple, batch_idx: int):
        *_, X, target = batch
        batch_size = X.size(0)
    
        # GPU に移動
        device = next(self.parameters()).device
        X = X.to(device)
        target = target.to(device)
    
        logits = self(X)["logits"]
        loss = F.cross_entropy(logits, target)
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))
        return batch_size, loss, acc1, acc5, logits

    def training_step(self, batch, batch_idx):
        self.backbone.train()  # ★ここをeval()からtrain()に
        _, loss, acc1, acc5, _ = self.shared_step(batch, batch_idx)
        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log, on_epoch=True, sync_dist=True, logger=True)
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
        if self.trainer.sanity_checking:
            return
    
        # 現タスク全体の集計
        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")
        self.log_dict({"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5},
                      on_epoch=True, sync_dist=True)
    
        # 現タスク・過去タスク別評価
        preds = torch.cat([o["logits"].max(-1)[1] for o in outs]).cpu().numpy()
        targets = torch.cat([o["targets"] for o in outs]).cpu().numpy()
        mask_correct = preds == targets
    
        tasks = getattr(self.hparams, "tasks", None)
        current_task_idx = getattr(self.hparams, "task_idx", 0)
    
        if tasks is not None:
            for task_idx, task in enumerate(tasks):
                if task_idx > current_task_idx:
                    continue
                mask_task = np.isin(targets, np.array(task))
                if mask_task.sum() == 0:
                    continue
                acc_task = np.logical_and(mask_task, mask_correct).sum() / mask_task.sum()
                # 個別にログして CSV に残す
                self.log(f"val_acc1_task{task_idx}", float(acc_task),
                         on_epoch=True, sync_dist=True)
    
        # 過去タスク累積評価
        if hasattr(self, "past_task_loaders") and self.past_task_loaders:
            for task_idx, loader in enumerate(self.past_task_loaders):
                if task_idx > current_task_idx:
                    continue
                preds_list, targets_list = [], []
                for batch in loader:
                    _, _, _, _, logits = self.shared_step(batch, 0)
                    preds_list.append(logits.argmax(dim=1).cpu().numpy())
                    targets_list.append(batch[-1].cpu().numpy())
                if not preds_list:
                    continue
                preds_past = np.concatenate(preds_list)
                targets_past = np.concatenate(targets_list)
                cum_acc = (preds_past == targets_past).sum() / len(targets_past)
                # 個別にログして CSV に残す
                self.log(f"cum_acc_task{task_idx}", float(cum_acc),
                         on_epoch=True, sync_dist=True)

    
    
    def evaluate_past_tasks(self):
        if not hasattr(self, "past_task_loaders") or not self.past_task_loaders:
            print("[INFO] No past tasks to evaluate.")
            return
    
        all_logs = []
        current_task_idx = getattr(self.hparams, "task_idx", 0)
    
        for task_idx, loader in enumerate(self.past_task_loaders):
            if task_idx > current_task_idx:
                continue
    
            preds_list, targets_list = [], []
            try:
                for batch in loader:
                    _, _, _, _, logits = self.shared_step(batch, 0)
                    preds_list.append(logits.argmax(dim=1).cpu().numpy())
                    targets_list.append(batch[-1].cpu().numpy())
            except Exception as e:
                print(f"[WARN] Skipping Task {task_idx} due to error: {e}")
                continue
    
            if not preds_list:
                continue
    
            preds = np.concatenate(preds_list)
            targets = np.concatenate(targets_list)
            acc = float((preds == targets).sum() / len(targets))
    
            print(f"[INFO] Validation accuracy for Task {task_idx}: {acc:.4f}")
    
            # self.log(...) はやめる
            all_logs.append({
                f"val_acc1_task{task_idx}": acc,
                f"cum_acc_task{task_idx}": acc,
            })
    
        if not all_logs:
            print("[INFO] No evaluation logs to record.")
            return
    
        # logger 経由で time-series として記録（任意）
        logger = getattr(self, "logger", None)
        if isinstance(logger, WandbLogger):
            # まとめて1ステップ分としてログ
            merged = {}
            for d in all_logs:
                merged.update(d)
            logger.experiment.log(merged, step=self.global_step)
    
        # 2-2) summary にも反映（Runs Table / CSV に必ず出る）
        import wandb
        if wandb.run is not None:
            for d in all_logs:
                for k, v in d.items():
                    wandb.run.summary[k] = v
            print("[INFO] WandB summary updated — values will appear in Runs Table.")
