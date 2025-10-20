# main_pretrain.py (修正版)

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path, PosixPath
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from kaizen.methods.linear_tpn import LinearTPNModel
from kaizen.methods import TPNLightning
from kaizen.args.setup import parse_args_pretrain
from kaizen.utils.pretrain_dataloader_wisdm import prepare_wisdm_dataloaders

# -----------------------------
# タスクマッピング
# -----------------------------
def map_labels_to_tasks():
    task_classes = [
        [17, 2, 0],    # タスク0
        [3, 10, 4],    # タスク1
        [4, 11, 16],   # タスク2
        [6, 1, 9],     # タスク3
        [7, 15, 5],    # タスク4
        [13, 12, 14],  # タスク5
    ]
    label_to_task = {}
    for idx, labels in enumerate(task_classes):
        for label in labels:
            label_to_task[label] = idx
    return task_classes, label_to_task

# -----------------------------
# データローダ準備（リプレイ含む）
# -----------------------------
def prepare_task_datasets(data_dir, task_idx, tasks, batch_size=64, num_workers=2, replay=False, replay_proportion=0.01):
    train_loader, val_loader = prepare_wisdm_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=0.1,
        num_workers=num_workers
    )

    loaders = {f"task{task_idx}": train_loader}

    # replay 用データセット
    if replay and task_idx != 0:
        replay_size = max(1, int(len(train_loader.dataset) * replay_proportion))
        indices = np.random.choice(len(train_loader.dataset), replay_size, replace=False)
        replay_dataset = Subset(train_loader.dataset, indices)
        replay_loader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        loaders["replay"] = replay_loader

    return loaders, val_loader

# -----------------------------
# 再現性の確保
# -----------------------------
def set_seed(seed=5):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# メイン処理
# -----------------------------
def main():
    args = parse_args_pretrain()
    SEED = getattr(args, "global_seed", 5)
    set_seed(SEED)
    seed_everything(SEED)

    # checkpoint_dir の存在確認
    if not hasattr(args, "checkpoint_dir") or args.checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be provided")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # タスク設定
    tasks, label_to_task = map_labels_to_tasks()
    task_idx = getattr(args, "task_idx", 0)

    # 過去タスク用データローダを保持
    past_task_loaders = []
    for prev_task_idx in range(task_idx):
        prev_loaders, _ = prepare_task_datasets(
            data_dir=args.data_dir,
            task_idx=prev_task_idx,
            tasks=tasks,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            replay=False
        )
        past_task_loaders.append(prev_loaders[f"task{prev_task_idx}"])

    # データ準備（現在タスク）
    train_loaders, val_loader = prepare_task_datasets(
        data_dir=args.data_dir,
        task_idx=task_idx,
        tasks=tasks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        replay=args.replay,
        replay_proportion=args.replay_proportion
    )

    # -----------------------------
    # TPNLightning: 特徴量抽出
    # -----------------------------
    backbone = TPNLightning(
        in_channels=3,
        feature_dim=getattr(args, "feature_dim", 128),
        num_classes=18,
        lr=args.lr if hasattr(args, "lr") else 1e-3
    )

    current_tpn_ckpt = os.path.join(args.checkpoint_dir, f"task{task_idx}_tpn.ckpt")
    if task_idx != 0:
        prev_tpn_ckpt = os.path.join(args.checkpoint_dir, f"task{task_idx-1}_tpn.ckpt")
        if os.path.exists(prev_tpn_ckpt):
            print(f"[INFO] Loading previous TPNLightning checkpoint from {prev_tpn_ckpt}")
            backbone.load_state_dict(torch.load(prev_tpn_ckpt, map_location="cpu")["state_dict"])
        else:
            print(f"[WARN] Previous TPN checkpoint not found, starting fresh.")

    tpn_trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        precision=args.precision if torch.cuda.is_available() else 32
    )
    tpn_trainer.fit(backbone, train_dataloaders=train_loaders[f"task{task_idx}"])
    tpn_trainer.save_checkpoint(current_tpn_ckpt)
    print(f"[INFO] Saved TPN checkpoint: {current_tpn_ckpt}")

    # -----------------------------
    # LinearTPN: 簡易評価 + 過去タスク累積評価
    # -----------------------------
    args_dict = vars(args).copy()
    args_dict.pop("num_classes", None)
    linear_model = LinearTPNModel(
        backbone=backbone,
        num_classes=18,
        past_task_loaders=past_task_loaders,
        **args_dict
    )

    linear_model.hparams["tasks"] = tasks
    linear_model.hparams["split_strategy"] = "class"
    linear_model.hparams["task_idx"] = task_idx

    callbacks = []
    wandb_logger = None
    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.name}_task{task_idx}",
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            mode="offline" if args.offline else "online",
            reinit=True,
        )
        wandb_logger.watch(linear_model, log="all", log_freq=100)
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    linear_ckpt = os.path.join(args.checkpoint_dir, f"task{task_idx}_last.ckpt")
    if args.save_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"task{task_idx}_last",
            save_last=True,
            save_top_k=1,
            monitor="train_loss",
            verbose=True
        )
        callbacks.append(checkpoint_callback)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        precision=args.precision if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    trainer.fit(linear_model,
            train_dataloaders=train_loaders[f"task{task_idx}"],
            val_dataloaders=val_loader) 

    # -----------------------------
    # 過去タスクを含めた累積評価
    # -----------------------------
    linear_model.evaluate_past_tasks()

    # -----------------------------
    # checkpoint保存 & last_checkpoint.txt 更新
    # -----------------------------
    if args.save_checkpoint:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        trainer.save_checkpoint(linear_ckpt)
        print(f"[INFO] Saved LinearTPN checkpoint to {linear_ckpt}")

        last_ckpt_txt = os.path.join(args.checkpoint_dir, "last_checkpoint.txt")
        with open(last_ckpt_txt, "w") as f:
            f.write(linear_ckpt)
        print(f"[INFO] Updated last_checkpoint.txt at {last_ckpt_txt}")

        # args.json 保存
        with open(os.path.join(args.checkpoint_dir, "args.json"), "w") as f:
            json.dump(
                {k: str(v) if isinstance(v, (Path, PosixPath)) else v for k, v in vars(args).items()},
                f,
                indent=4
            )

if __name__ == "__main__":
    main()
