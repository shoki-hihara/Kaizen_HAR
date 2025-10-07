import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from kaizen.args.setup import parse_args_eval
from kaizen.utils.checkpointer import Checkpointer
from tpn_model import TPN
from linear_tpn_model import LinearTPNModel
from kaizen.nethods.dataloader import load_wisdm_dataset  # 独自データセット

def main():
    args = parse_args_eval()
    seed_everything(args.global_seed)

    # -------------------------------
    # データセット読み込み (HAR)
    # -------------------------------
    train_dataset, val_dataset = load_wisdm_dataset(args.data_dir)

    # -------------------------------
    # 固定タスク分割
    # -------------------------------
    tasks = [
        [0, 1, 2],      # タスク1
        [3, 4, 5],      # タスク2
        [6, 7, 8],      # タスク3
        [9, 10, 11],    # タスク4
        [12, 13, 14],   # タスク5
        [15, 16, 17],   # タスク6
    ]
    task_idx = args.task_idx
    task_classes = tasks[task_idx]

    task_indices = [i for i, label in enumerate(train_dataset.labels) if label in task_classes]
    task_dataset = Subset(train_dataset, task_indices)
    task_loader = DataLoader(
        task_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    train_loaders = {f"task{task_idx}": task_loader}

    # -------------------------------
    # Replay データ
    # -------------------------------
    if args.replay and task_idx != 0:
        replay_indices = []
        for prev_idx in range(task_idx):
            prev_classes = tasks[prev_idx]
            replay_indices.extend([i for i, label in enumerate(train_dataset.labels) if label in prev_classes])
        replay_dataset = Subset(train_dataset, replay_indices)
        replay_loader = DataLoader(
            replay_dataset,
            batch_size=min(args.replay_batch_size, len(replay_dataset)),
            shuffle=True,
            num_workers=args.num_workers
        )
        train_loaders.update({"replay": replay_loader})

    # -------------------------------
    # Online evaluation
    # -------------------------------
    if args.online_evaluation:
        if args.online_evaluation_training_data_source == "all_tasks":
            online_eval_indices = list(range(len(train_dataset)))
        elif args.online_evaluation_training_data_source == "current_task":
            online_eval_indices = task_indices
        elif args.online_evaluation_training_data_source == "seen_tasks":
            online_eval_indices = []
            for i in range(task_idx + 1):
                seen_classes = tasks[i]
                online_eval_indices.extend([j for j, label in enumerate(train_dataset.labels) if label in seen_classes])
        else:
            raise ValueError(f"Unknown online_evaluation_training_data_source: {args.online_evaluation_training_data_source}")

        online_eval_dataset = Subset(train_dataset, online_eval_indices)
        online_eval_loader = DataLoader(
            online_eval_dataset,
            batch_size=-(-len(online_eval_dataset) // len(task_loader)),  # ceil division
            shuffle=False,
            num_workers=args.num_workers
        )
        train_loaders.update({"online_eval": online_eval_loader})

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # -------------------------------
    # モデル構築: TPN + LinearTPNModel
    # -------------------------------
    backbone = TPN(in_channels=train_dataset.num_channels, feature_dim=args.feature_dim)
    model = LinearTPNModel(backbone=backbone, num_classes=args.num_classes)

    # -------------------------------
    # 事前学習済みモデルのロード
    # -------------------------------
    if args.pretrained_model:
        assert args.pretrained_model.endswith((".ckpt", ".pth", ".pt"))
        state = torch.load(args.pretrained_model, map_location=None if torch.cuda.is_available() else "cpu")["state_dict"]
        extracted_state = {k.replace("backbone.", ""): v for k, v in state.items() if "backbone" in k}
        missing_keys, unexpected_keys = backbone.load_state_dict(extracted_state, strict=False)
        print("Missing keys - Backbone:", missing_keys)
        print("Unexpected keys - Backbone:", unexpected_keys)

        # classifier_eval / online_classifier_eval 用の重みロード
        if args.evaluation_mode in ["classifier_eval", "online_classifier_eval"]:
            extracted_state = {k.replace("classifier.", ""): v for k, v in state.items() if "classifier" in k}
            missing_keys_cls, unexpected_keys_cls = model.classifier.load_state_dict(extracted_state, strict=False)
            print("Missing keys - Classifier:", missing_keys_cls)
            print("Unexpected keys - Classifier:", unexpected_keys_cls)

    # -------------------------------
    # Trainer / Callbacks
    # -------------------------------
    callbacks = []

    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.name}-task{task_idx}",
            project=args.project,
            entity=args.entity,
            offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        if args.save_checkpoint:
            ckpt = Checkpointer(
                args,
                logdir=os.path.join(args.checkpoint_dir, "eval"),
                frequency=args.checkpoint_frequency,
            )
            callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        checkpoint_callback=False,
        terminate_on_nan=True
    )

    # -------------------------------
    # 実行: 評価 or 学習
    # -------------------------------
    if args.evaluation_mode == "linear_eval":
        trainer.fit(model, train_loaders, val_loader)
    else:
        trainer.validate(model, val_loader)


if __name__ == "__main__":
    main()
