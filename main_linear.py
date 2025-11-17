import os
import types

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models import resnet18, resnet50

from kaizen.args.setup import parse_args_linear

try:
    from kaizen.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

from kaizen.methods.linear import LinearModel
from kaizen.utils.classification_dataloader import prepare_data
from kaizen.utils.checkpointer import Checkpointer

# ===== HAR / TPN 用の import =====
from kaizen.methods.tpn import TPN
from kaizen.methods.linear_tpn import LinearTPNModel
from kaizen.utils.har_dataset_utils import prepare_task_datasets

# ===== wisdm2019 用 固定タスク =====
def _fixed_wisdm_tasks():
    task_classes = [
        [17, 2, 0],    # タスク0
        [3, 10, 4],    # タスク1
        [4, 11, 16],   # タスク2
        [6, 1, 9],     # タスク3
        [7, 15, 5],    # タスク4
        [13, 12, 14],  # タスク5
    ]
    import torch as _torch
    return tuple(_torch.tensor(cls, dtype=_torch.long) for cls in task_classes)


def main():
    args = parse_args_linear()

    if not hasattr(args, "replay"):
        args.replay = False
    if not hasattr(args, "replay_proportion"):
        args.replay_proportion = 0.0
    if not hasattr(args, "replay_batch_size"):
        args.replay_batch_size = 0

    # ==========================
    # 0️⃣ タスク分割の設定
    # ==========================
    tasks = None
    if args.split_strategy == "class":
        if args.dataset.lower() == "wisdm2019":
            # HAR は pretrain 側と同じ固定タスク
            tasks = _fixed_wisdm_tasks()
        else:
            # 画像系は従来どおりランダムクラス分割
            assert args.num_classes % args.num_tasks == 0
            torch.manual_seed(args.split_seed)
            tasks = torch.randperm(args.num_classes).chunk(args.num_tasks)

    seed_everything(args.global_seed)

    # ==========================
    # 1️⃣ HAR (wisdm2019 + TPN) 用ブランチ
    # ==========================
    if args.dataset.lower() == "wisdm2019" and args.encoder == "tpn":
        # --- TPN バックボーン構築 ---
        tpn_backbone = TPN(
            input_dim=args.input_dim,
            feature_dim=args.feature_dim,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )

        # --- 事前学習済み TPN 重みをロード ---
        assert (
            args.pretrained_feature_extractor.endswith(".ckpt")
            or args.pretrained_feature_extractor.endswith(".pth")
            or args.pretrained_feature_extractor.endswith(".pt")
        ), f"unexpected checkpoint extension: {args.pretrained_feature_extractor}"
        ckpt_path = args.pretrained_feature_extractor
        assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"

        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]

        new_state = {}
        for k, v in state.items():
            # encoder/backbone だけ取り出して TPN にマッピング
            if k.startswith("encoder.") or k.startswith("backbone."):
                new_k = k.replace("encoder.", "").replace("backbone.", "")
                new_state[new_k] = v
        missing, unexpected = tpn_backbone.load_state_dict(new_state, strict=False)
        print(f"Loaded {ckpt_path}, missing={missing}, unexpected={unexpected}")

        # --- HAR 用タスク別データローダ ---
        task_idx = getattr(args, "task_idx", 0)
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

        # 過去タスク用 loader 一覧（累積 Acc 用）
        past_task_loaders = [test_loaders[f"task{i}"] for i in range(task_idx + 1)]

        replay = getattr(args, "replay", False)
        replay_proportion = getattr(args, "replay_proportion", 0.0)
        replay_batch_size = getattr(args, "replay_batch_size", 0)

        model_kwargs = vars(args).copy()
        model_kwargs.pop("num_classes", None)

        # --- LinearTPNModel 構築 ---
        model = LinearTPNModel(
            backbone=tpn_backbone,
            num_classes=args.num_classes,
            past_task_loaders=past_task_loaders,
            tasks=tasks,
            freeze_backbone=True,   # 純粋な線形評価
            **model_kwargs,
        )

        # --- コールバック / ロガー ---
        callbacks = []
        if args.wandb:
            wandb_logger = WandbLogger(
                name=args.name,
                project=args.project,
                entity=args.entity,
                offline=getattr(args, "offline", False),
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

        trainer = Trainer.from_argparse_args(
            args,
            logger=wandb_logger,
            callbacks=callbacks,
            plugins=DDPPlugin(find_unused_parameters=True),
            checkpoint_callback=False,
            terminate_on_nan=True,
            accelerator="ddp",
        )

        print(f"🚀 Start Linear Evaluation for WISDM Task {task_idx}")
        trainer.fit(model, train_loader, val_loader)
        print(f"✅ Finished Linear Eval for WISDM Task {task_idx}")
        return

    # ==========================
    # 2️⃣ 元の画像用ブランチ（ResNet + LinearModel）
    # ==========================
    if args.encoder == "resnet18":
        backbone = resnet18()
    elif args.encoder == "resnet50":
        backbone = resnet50()
    else:
        raise ValueError("Only [resnet18, resnet50] are currently supported.")

    if args.cifar:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()

    assert (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    )
    ckpt_path = args.pretrained_feature_extractor

    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(state, strict=False)

    print(f"Loaded {ckpt_path}")

    if args.dali:
        assert _dali_avaliable, "Dali is not currently avaiable, please install it first."
        MethodClass = types.new_class(
            f"Dali{LinearModel.__name__}", (ClassificationABC, LinearModel)
        )
    else:
        MethodClass = LinearModel

    model = MethodClass(backbone, **args.__dict__, tasks=tasks)

    train_loader, val_loader = prepare_data(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        semi_supervised=args.semi_supervised,
    )

    callbacks = []

    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, "linear"),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    else:
        wandb_logger = None

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator="ddp",
    )
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
