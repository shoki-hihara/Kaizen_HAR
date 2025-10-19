# main_pretrain.py 完全修正版
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from kaizen.methods import TPN, LinearTPNModel
from kaizen.args.setup import parse_args_pretrain
from kaizen.utils.pretrain_dataloader_wisdm import prepare_wisdm_dataloaders

def map_labels_to_tasks():
    task_classes = [
        [17, 2, 0],    # タスク1
        [3, 10, 4],    # タスク2
        [4, 11, 16],   # タスク3
        [6, 1, 9],     # タスク4
        [7, 15, 5],    # タスク5
        [13, 12, 14],  # タスク6
    ]
    label_to_task = {}
    for idx, labels in enumerate(task_classes):
        for label in labels:
            label_to_task[label] = idx
    return task_classes, label_to_task

def prepare_task_datasets(data_dir, task_idx, tasks, batch_size=64, num_workers=2, replay=False, replay_proportion=0.01):
    train_loader, val_loader = prepare_wisdm_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=0.1,
        num_workers=num_workers
    )

    train_loaders = {f"task{task_idx}": train_loader}

    if replay and task_idx != 0:
        replay_sample_size = max(1, int(len(train_loader.dataset) * replay_proportion))
        indices = np.random.choice(len(train_loader.dataset), size=replay_sample_size, replace=False)
        replay_dataset = torch.utils.data.Subset(train_loader.dataset, indices)
        replay_loader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        train_loaders["replay"] = replay_loader

    return train_loaders, val_loader

def set_seed(seed: int = 5):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args_pretrain()

    # -----------------------------
    # Seed 設定
    # -----------------------------
    SEED = getattr(args, "global_seed", 5)
    set_seed(SEED)
    seed_everything(SEED)

    # -----------------------------
    # タスクマッピング
    # -----------------------------
    tasks, label_to_task = map_labels_to_tasks()

    # -----------------------------
    # データセット準備
    # -----------------------------
    train_loaders, val_loader = prepare_task_datasets(
        data_dir=args.data_dir,
        task_idx=args.task_idx,
        tasks=tasks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        replay=args.replay,
        replay_proportion=args.replay_proportion
    )

    print(f"[DEBUG] Loaded train loaders: {[len(dl.dataset) for dl in train_loaders.values()]}")

    # -----------------------------
    # モデル構築
    # -----------------------------
    feature_dim = getattr(args, "feature_dim", 128)
    backbone = TPN(in_channels=3, feature_dim=feature_dim)
    model = LinearTPNModel(backbone=backbone, num_classes=len(sum(tasks, [])))

    # -----------------------------
    # WandB ログ
    # -----------------------------
    callbacks = []
    wandb_logger = None
    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.name}-task{args.task_idx}",
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            reinit=True,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # -----------------------------
    # Checkpoint
    # -----------------------------
    if args.save_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="last",
            save_last=True,
            save_top_k=1,
            monitor="val_loss",  # <<< ReduceLROnPlateau 対応
            verbose=True
        )
        callbacks.append(checkpoint_callback)

    # -----------------------------
    # GPU 判定
    # -----------------------------
    accelerator = "gpu" if torch.cuda.is_available() and int(args.gpus) > 0 else "cpu"
    devices = 1 if accelerator == "gpu" else None
    precision = args.precision if accelerator == "gpu" else 32

    # -----------------------------
    # Task 1 以降 checkpoint からロード
    # -----------------------------
    last_ckpt_path = os.path.join(args.checkpoint_dir, "last.ckpt")
    if args.task_idx != 0:
        if not os.path.exists(last_ckpt_path):
            raise FileNotFoundError(f"Task 0 checkpoint not found at {last_ckpt_path}")
        print(f"[INFO] Loading checkpoint from {last_ckpt_path} for Task {args.task_idx}")
        model = LinearTPNModel.load_from_checkpoint(
            last_ckpt_path, backbone=model.backbone, num_classes=len(sum(tasks, []))
        )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=wandb_logger
    )

    # -----------------------------
    # 学習開始
    # -----------------------------
    trainer.fit(
        model,
        train_dataloaders=train_loaders[f"task{args.task_idx}"],
        val_dataloaders=val_loader
    )

    # -----------------------------
    # Task 0 checkpoint 保存
    # -----------------------------
    if args.task_idx == 0 and args.save_checkpoint:
        trainer.save_checkpoint(last_ckpt_path)
        print(f"[INFO] Checkpoint saved to {last_ckpt_path}")

if __name__ == "__main__":
    main()
