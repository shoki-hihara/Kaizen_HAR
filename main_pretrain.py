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

from kaizen.methods.dataloader import load_wisdm_dataset, get_dataloader
from kaizen.methods import TPN
from kaizen.methods import LinearTPNModel

from kaizen.args.setup import parse_args_pretrain
from kaizen.methods import METHODS
from kaizen.distillers import DISTILLERS
from kaizen.distiller_factories import DISTILLER_FACTORIES, base_frozen_model_factory
from kaizen.utils.checkpointer import Checkpointer

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
    for task_idx, labels in task_classes.items():
        for label in labels:
            label_to_task[label] = task_idx
    return task_classes, label_to_task

def prepare_task_datasets(data_dir, task_idx, tasks, batch_size=64, num_workers=2, replay=False, replay_proportion=0.01):
    """
    HARデータセットのタスク分割、replay, online_eval対応
    """
    # 全データセット
    train_dataset = load_wisdm_dataset(data_dir, split="train")
    test_dataset = load_wisdm_dataset(data_dir, split="test")

    # タスクごとのラベル
    task_labels = tasks[task_idx]

    # タスクのサブセット
    task_indices = [i for i, (_, y) in enumerate(train_dataset) if y in task_labels]
    task_dataset = Subset(train_dataset, task_indices)
    task_loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    train_loaders = {f"task{task_idx}": task_loader}

    # replay データ
    if replay and task_idx != 0:
        replay_indices = [i for i, (_, y) in enumerate(train_dataset) if any(y in tasks[t] for t in range(task_idx))]
        replay_sample_size = max(1, int(len(replay_indices) * replay_proportion))
        replay_indices = np.random.choice(replay_indices, size=replay_sample_size, replace=False)
        replay_dataset = Subset(train_dataset, replay_indices)
        replay_loader = DataLoader(replay_dataset, batch_size=min(batch_size, len(replay_dataset)),
                                   shuffle=True, num_workers=num_workers)
        train_loaders.update({"replay": replay_loader})

    # online_eval データ
    online_eval_indices = [i for i, (_, y) in enumerate(test_dataset) if y in sum(tasks[:task_idx + 1], [])]
    online_eval_dataset = Subset(test_dataset, online_eval_indices)
    online_eval_loader = DataLoader(online_eval_dataset, batch_size=len(online_eval_dataset) // len(task_loader),
                                    shuffle=False, num_workers=num_workers)
    train_loaders.update({"online_eval": online_eval_loader})

    return train_loaders, test_dataset

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

    print(f"[DEBUG] Loaded train loaders: {[len(dl.dataset) for dl in train_loaders]}")

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
        ckpt = Checkpointer(
            args,
            logdir=args.checkpoint_dir,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=callbacks,
        checkpoint_callback=False,
        terminate_on_nan=True,
    )

    trainer.fit(model, train_loaders, val_dataloaders=None)

if __name__ == "__main__":
    main()
