import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path, PosixPath
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from kaizen.methods.linear_tpn import LinearTPNModel
from kaizen.methods import TPNLightning  # backbone 用
from kaizen.args.setup import parse_args_pretrain
from kaizen.utils.pretrain_dataloader_wisdm import prepare_wisdm_dataloaders

# -----------------------------
# タスクマッピング
# -----------------------------
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
# タスクループ実行
# -----------------------------
def main():
    args = parse_args_pretrain()
    SEED = getattr(args, "global_seed", 5)
    set_seed(SEED)
    seed_everything(SEED)

    # タスク設定
    tasks, label_to_task = map_labels_to_tasks()

    # backbone の初期化
    backbone = TPNLightning(
        in_channels=3,
        feature_dim=getattr(args, "feature_dim", 128),
        num_classes=18,  # dummy
        lr=args.lr if hasattr(args, "lr") else 1e-3
    )

    # task0 → taskN ループ
    for task_idx in range(len(tasks)):
        print(f"\n[INFO] ======== Task {task_idx} ========")

        # データローダ準備
        train_loaders, val_loader = prepare_task_datasets(
            data_dir=args.data_dir,
            task_idx=task_idx,
            tasks=tasks,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            replay=args.replay,
            replay_proportion=args.replay_proportion
        )

        # LinearTPN モデル構築（簡易評価用）
        args_dict = vars(args).copy()
        args_dict.pop("num_classes", None)
        model = LinearTPNModel(
            backbone=backbone,
            num_classes=18,
            eval_linear=True,
            **args_dict
        )

        # -----------------------------
        # 前タスクの重みを引き継ぎ
        # -----------------------------
        current_ckpt = os.path.join(args.checkpoint_dir, f"task{task_idx}_last.ckpt")
        if task_idx != 0:
            prev_ckpt = os.path.join(args.checkpoint_dir, f"task{task_idx - 1}_last.ckpt")
            if os.path.exists(prev_ckpt):
                print(f"[INFO] Loading checkpoint from {prev_ckpt}")
                ckpt = torch.load(prev_ckpt, map_location="cpu")
                model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                print(f"[WARN] Previous checkpoint not found for task {task_idx - 1}, starting fresh.")

        # -----------------------------
        # WandB ロガー
        # -----------------------------
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
            wandb_logger.watch(model, log="all", log_freq=100)
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # -----------------------------
        # Checkpoint コールバック
        # -----------------------------
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

        # -----------------------------
        # Trainer 設定
        # -----------------------------
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1 if accelerator == "gpu" else None
        precision = args.precision if accelerator == "gpu" else 32

        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            logger=wandb_logger,
        )

        # -----------------------------
        # 学習開始（特徴量抽出 + LinearTPN）
        # -----------------------------
        trainer.fit(
            model,
            train_dataloaders=train_loaders[f"task{task_idx}"]
        )

        # -----------------------------
        # checkpoint保存 & last_checkpoint.txt 更新
        # -----------------------------
        if args.save_checkpoint:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            trainer.save_checkpoint(current_ckpt)
            print(f"[INFO] Saved checkpoint to {current_ckpt}")

            last_ckpt_txt = os.path.join(args.checkpoint_dir, "last_checkpoint.txt")
            with open(last_ckpt_txt, "w") as f:
                f.write(current_ckpt)
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
