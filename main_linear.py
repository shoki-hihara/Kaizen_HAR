# main_linear.py (TPN + HAR対応)
import os
import types
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from linear_tpn_model import LinearTPNModel
from tpn_model import TPN
from kaizen.methods.dataloader import load_wisdm_dataset
from har_dataset_utils import prepare_task_datasets
from kaizen.utils.checkpointer import Checkpointer
from kaizen.args.setup import parse_args_linear


def main():
    args = parse_args_linear()
    seed_everything(args.global_seed)

    # ===============================
    # 1️⃣ タスク分割の設定
    # ===============================
    tasks = None
    if args.split_strategy == "class":
        assert args.num_classes % args.num_tasks == 0
        torch.manual_seed(args.split_seed)
        tasks = torch.randperm(args.num_classes).chunk(args.num_tasks)
        tasks = [t.tolist() for t in tasks]

    # ===============================
    # 2️⃣ TPNバックボーンの初期化
    # ===============================
    tpn_backbone = TPN(
        input_dim=args.input_dim,
        feature_dim=args.feature_dim,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    # 事前学習済みTPN重みをロード
    if args.pretrained_feature_extractor:
        ckpt_path = args.pretrained_feature_extractor
        assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"
        state = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" in state:
            state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            if k.startswith("encoder.") or k.startswith("backbone."):
                new_k = k.replace("encoder.", "").replace("backbone.", "")
                new_state[new_k] = v
        missing, unexpected = tpn_backbone.load_state_dict(new_state, strict=False)
        print(f"Loaded {ckpt_path}, missing={missing}, unexpected={unexpected}")
    else:
        print("⚠️ No pretrained TPN weights loaded")

    # ===============================
    # 3️⃣ モデル構築（Linear分類器）
    # ===============================
    model = LinearTPNModel(
        backbone=tpn_backbone,
        num_classes=args.num_classes,
        **vars(args),
        tasks=tasks,
    )

    # ===============================
    # 4️⃣ データローダー構築 (HAR + タスク対応)
    # ===============================
    task_idx = args.task_idx if hasattr(args, "task_idx") else 0
    train_loaders, test_loaders = prepare_task_datasets(
        data_dir=args.data_dir,
        task_idx=task_idx,
        tasks=tasks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        replay=args.replay,
        replay_proportion=args.replay_proportion,
    )

    train_loader = train_loaders[f"task{task_idx}"]
    val_loader = test_loaders[f"task{task_idx}"]

    # ===============================
    # 5️⃣ WandB・Callback設定
    # ===============================
    callbacks = []

    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        ckpt_callback = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, f"linear_task{task_idx}"),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt_callback)
    else:
        wandb_logger = None

    # ===============================
    # 6️⃣ Trainer設定
    # ===============================
    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator="ddp",
    )

    # ===============================
    # 7️⃣ 学習実行
    # ===============================
    print(f"🚀 Start Linear Evaluation for Task {task_idx}")
    trainer.fit(model, train_loader, val_loader)
    print(f"✅ Finished Linear Eval for Task {task_idx}")


if __name__ == "__main__":
    main()
