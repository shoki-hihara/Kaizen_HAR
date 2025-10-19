# main_pretrain.py
import os
from pprint import pprint
import types
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from kaizen.methods.dataloader import load_wisdm_dataset, get_dataloader
from kaizen.methods import TPN
from kaizen.methods import LinearTPNModel

from kaizen.args.setup import parse_args_pretrain
from kaizen.methods import METHODS
from kaizen.distillers import DISTILLERS
from kaizen.distiller_factories import DISTILLER_FACTORIES, base_frozen_model_factory
from kaizen.utils.checkpointer import Checkpointer
from kaizen.utils.pretrain_dataloader_wisdm import prepare_wisdm_dataloaders

import random

def map_labels_to_tasks():
    """
    HARのタスク分割ラベルマッピング
    """
    task_classes = [
        [17, 2, 0],    # タスク1　"folding_clothes", "climb_stairs", "walking"
        [3, 10, 4],    # タスク2　"sitting", "drinking", "eating_fries"
        [4, 11, 16],   # タスク3　"standing", "eating_sandwich", "clapping_hands"
        [6, 1, 9],     # タスク4　"brushing_teeth", "jogging", "eating_pasta"
        [7, 15, 5],    # タスク5　"eating_soup", "writing", "typing"
        [13, 12, 14],  # タスク6　"catching_ball", "kicking_soccer", "dribbling_basketball"
    ]
    
    label_to_task = {}
    for idx, labels in enumerate(task_classes):
        for label in labels:
            label_to_task[label] = idx

    return task_classes, label_to_task

def prepare_task_datasets(data_dir, task_idx, tasks, batch_size=64, num_workers=2, replay=False, replay_proportion=0.01):
    """
    WISDM 用の train/val loader 構築
    """
    # データをwindow化してDataLoader作成
    train_loader, val_loader, label_encoder = prepare_wisdm_dataloaders(
        csv_path=os.path.join(data_dir, "WISDM_ar_v1.1_raw.txt"),
        batch_size=batch_size,
        window_size=384,
        step_size=384,
        val_ratio=0.2,
        num_workers=num_workers
    )

    train_loaders = {f"task{task_idx}": train_loader}

    # replayやonline_evalは必要に応じて追加可能
    if replay and task_idx != 0:
        # replayロジック（同じtrain_loaderからサンプル抽出）
        replay_sample_size = max(1, int(len(train_loader.dataset) * replay_proportion))
        indices = np.random.choice(len(train_loader.dataset), size=replay_sample_size, replace=False)
        replay_dataset = torch.utils.data.Subset(train_loader.dataset, indices)
        replay_loader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        train_loaders["replay"] = replay_loader

    # val_loader は trainer.fit の val_dataloaders に渡す
    return train_loaders, val_loader

def set_seed(seed: int = 5):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 再現性重視
    torch.backends.cudnn.benchmark = False     # 再現性重視

def main():
    SEED = 5
    set_seed(SEED)
    
    args = parse_args_pretrain()
    seed_everything(args.global_seed)

    # タスクラベルマッピング
    tasks, label_to_task = map_labels_to_tasks()

    # HAR データセット準備
    train_loaders, val_dataset = prepare_task_datasets(
        data_dir=args.data_dir,
        task_idx=args.task_idx,
        tasks=tasks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        replay=args.replay,
        replay_proportion=args.replay_proportion
    )

    print(f"[DEBUG] Loaded train loaders: {[len(dl.dataset) for dl in train_loaders.values()]}")

    # モデル構築
    feature_dim = getattr(args, "feature_dim", 128)
    backbone = TPN(in_channels=3, feature_dim=feature_dim)
    model = LinearTPNModel(backbone=backbone, num_classes=len(sum(tasks, [])))

    # WandB ログ
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
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    if args.save_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="last",
            save_last=True,
            save_top_k=1,
            monitor=None,
            verbose=True   # 保存状況をログに出力
        )
        callbacks.append(checkpoint_callback)

    gpu_arg = 0
    if isinstance(args.gpus, list):
        gpu_arg = int(args.gpus[0])
    elif isinstance(args.gpus, str):
        gpu_arg = int(args.gpus)
    
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if gpu_arg > 0 else "cpu",
        devices=1 if gpu_arg > 0 else None,   # ← ここを修正
        precision=args.precision,
        callbacks=callbacks,
        logger=wandb_logger if args.wandb else None,
    )

    last_ckpt_path = os.path.join(args.checkpoint_dir, "last.ckpt")
    
    # Task 1 以降でのみ checkpoint をロード
    last_ckpt_path = os.path.join(args.checkpoint_dir, "last.ckpt")
    if args.task_idx != 0:
        if not os.path.exists(last_ckpt_path):
            raise FileNotFoundError(f"Task 0 checkpoint not found at {last_ckpt_path}")
        print(f"[INFO] Loading checkpoint from {last_ckpt_path} for Task {args.task_idx}")
        model = LinearTPNModel.load_from_checkpoint(
            last_ckpt_path, backbone=model.backbone, num_classes=len(sum(tasks, []))
        )

    print(f"[DEBUG] Training on {len(train_loaders[f'task{args.task_idx}'].dataset)} samples")
    print(f"[DEBUG] Checkpoint will be saved to: {args.checkpoint_dir}")
    
    trainer.fit(
        model,
        train_loaders[f"task{args.task_idx}"],
        val_dataloaders=val_dataset  # 上で返したval_loader
    )

if __name__ == "__main__":
    main()
