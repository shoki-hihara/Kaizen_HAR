# main_pretrain.py (HAR対応 + 固定タスク + 蒸留維持)

from ast import arg
import os
from pprint import pprint
import types

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from kaizen.args.setup import parse_args_pretrain
from kaizen.methods import METHODS
from kaizen.distillers import DISTILLERS
from kaizen.distiller_factories import DISTILLER_FACTORIES, base_frozen_model_factory

# DALI (未対応: 大元準拠)
try:
    from kaizen.methods.dali import PretrainABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

# UMAP (任意)
try:
    from kaizen.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

from kaizen.utils.checkpointer import Checkpointer
from kaizen.utils.classification_dataloader import prepare_data as prepare_data_classification
from kaizen.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_multicrop_transform,
    prepare_n_crop_transform,
    prepare_transform,
    split_dataset,
    split_dataset_subset,
    direct_prepare_split_dataset_subset,
)


# --- 固定タスク分割（wisdm2019用） ---
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
    args = parse_args_pretrain()

    # DALIは未対応（大元と同じ方針）
    if args.dali:
        raise NotImplementedError("Dali is not supported")

    # Online evaluation の DataLoader 複数切替ポリシー
    args.multiple_trainloader_mode = "max_size_cycle"  # 大元値

    # Online eval のバッチサイズ
    args.online_eval_batch_size = int(args.batch_size) if args.online_evaluation else None

    # --- タスク分割（class-split） ---
    tasks = None
    if args.split_strategy == "class":
        if getattr(args, "dataset", "").lower() == "wisdm2019":
            # 固定タスクを使用
            assert args.num_classes == 18, f"wisdm2019 expected num_classes=18, got {args.num_classes}"
            assert args.num_tasks == 6, f"wisdm2019 expected num_tasks=6, got {args.num_tasks}"
            tasks = _fixed_wisdm_tasks()
        else:
            assert args.num_classes % args.num_tasks == 0
            torch.manual_seed(args.split_seed)
            tasks = torch.randperm(args.num_classes).chunk(args.num_tasks)
        print("Task Classes (resolved):", tasks)

    # 乱数固定
    seed_everything(args.global_seed)

    # --- データ準備（大元ロジックを踏襲） ---
    if not args.dali:
        # 変換の構築
        if args.unique_augs > 1:
            transform = [
                prepare_transform(args.dataset, multicrop=args.multicrop, **kwargs)
                for kwargs in args.transform_kwargs
            ]
        else:
            transform = prepare_transform(
                args.dataset, multicrop=args.multicrop, **args.transform_kwargs
            )

        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        # マルチクロップかどうか
        if args.multicrop:
            assert not args.unique_augs == 1
            if args.dataset in ["cifar10", "cifar100", "wisdm2019"]:
                size_crops = [32, 24]
            elif args.dataset == "stl10":
                size_crops = [96, 58]
            else:
                size_crops = [224, 96]
            transform = prepare_multicrop_transform(
                transform,
                size_crops=size_crops,
                num_crops=[args.num_crops, args.num_small_crops],
            )
        else:
            if args.num_crops != 2:
                assert args.method == "wmse"
            online_eval_transform = transform[-1] if isinstance(transform, list) else transform
            task_transform = prepare_n_crop_transform(transform, num_crops=args.num_crops)

        # データセット
        train_dataset, online_eval_dataset = prepare_datasets(
            args.dataset,
            task_transform=task_transform,
            online_eval_transform=online_eval_transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            no_labels=args.no_labels,
        )

        # タスク分割
        task_dataset, tasks = split_dataset(
            train_dataset,
            tasks=tasks,
            task_idx=args.task_idx,
            num_tasks=args.num_tasks,
            split_strategy=args.split_strategy,
            split_seed=args.split_seed,
        )

        # データローダ
        task_loader = prepare_dataloader(
            task_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train_loaders = {f"task{args.task_idx}": task_loader}

        # リプレイ
        if args.replay and args.task_idx != 0:
            replay_dataset = direct_prepare_split_dataset_subset(
                dataset=args.dataset,
                task_transform=task_transform,
                online_eval_transform=online_eval_transform,
                data_dir=args.data_dir,
                train_dir=args.train_dir,
                no_labels=args.no_labels,
                tasks=tasks,
                replay_task_idxs=np.arange(args.task_idx),
                num_tasks=args.num_tasks,
                split_strategy=args.split_strategy,
                split_seed=args.split_seed,
                proportion=args.replay_proportion,
                num_samples=args.replay_memory_bank_size,
            )
            replay_loader = prepare_dataloader(
                replay_dataset,
                batch_size=min(args.replay_batch_size, len(replay_dataset)),
                num_workers=args.num_workers,
            )
            train_loaders.update({"replay": replay_loader})

        # Online eval
        if args.online_eval_batch_size:
            if args.online_evaluation_training_data_source == "all_tasks":
                online_eval_dataset_final = online_eval_dataset
            elif args.online_evaluation_training_data_source == "current_task":
                online_eval_dataset_final, _ = split_dataset(
                    online_eval_dataset,
                    tasks=tasks,
                    task_idx=args.task_idx,
                    num_tasks=args.num_tasks,
                    split_strategy=args.split_strategy,
                    split_seed=args.split_seed,
                )
            elif args.online_evaluation_training_data_source == "seen_tasks":
                task_idxs = [i for i in range(args.task_idx + 1)]
                online_eval_dataset_final, _ = split_dataset(
                    online_eval_dataset,
                    tasks=tasks,
                    task_idx=task_idxs,
                    num_tasks=args.num_tasks,
                    split_strategy=args.split_strategy,
                    split_seed=args.split_seed,
                )
            else:
                online_eval_dataset_final = online_eval_dataset

            online_eval_loader = prepare_dataloader(
                online_eval_dataset_final,
                batch_size=-(-len(online_eval_dataset_final) // len(task_loader)),
                num_workers=args.num_workers,
            )
            train_loaders.update({"online_eval": online_eval_loader})

        # --- 検証用データローダ ---
        if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
            val_loader = None
        elif args.dataset in ["imagenet100", "imagenet"] and args.val_dir is None:
            val_loader = None
        elif args.dataset == "wisdm2019":
            # HAR 用 pretrain では画像用の分類 val_loader は使わない
            val_loader = None
        else:
            _, val_loader = prepare_data_classification(
                args.dataset,
                data_dir=args.data_dir,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=2 * args.num_workers,
            )

    # --- 手法クラス構築 ---
    assert args.method in METHODS, f"Choose from {METHODS.keys()}"
    MethodClass = METHODS[args.method]

    if args.dali:
        assert _dali_avaliable, "Dali is not currently available; install with [dali]."
        MethodClass = types.new_class(f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass))

    # --- 蒸留の有無（Kaizen準拠） ---
    if args.distiller_library == "default":
        if args.distiller:
            MethodClass = DISTILLERS[args.distiller](MethodClass)
    elif args.distiller_library == "factory":
        if args.distiller:
            MethodClass = base_frozen_model_factory(MethodClass)
            MethodClass = DISTILLER_FACTORIES[args.distiller](
                MethodClass,
                distill_current_key="z",
                distill_frozen_key="frozen_z",
                output_dim=args.output_dim,
                class_tag="feature_extractor",
            )
            if args.classifier_training and args.distiller_classifier is not None:
                MethodClass = DISTILLER_FACTORIES[args.distiller_classifier](
                    MethodClass,
                    distill_current_key="classifier_logits",
                    distill_frozen_key="frozen_logits",
                    output_dim=args.num_classes,
                    class_tag="classifier",
                )

    # --- モデル生成 ---
    model = MethodClass(**args.__dict__, tasks=tasks if args.split_strategy == "class" else None)

    # --- チェックポイント再開処理 ---
    assert [args.resume_from_checkpoint, args.pretrained_model].count(True) <= 1
    if args.resume_from_checkpoint:
        pass
    elif args.pretrained_model:
        print(f"Loading previous task checkpoint {args.pretrained_model}...")
        state_dict = torch.load(args.pretrained_model, map_location="cpu")["state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    # --- コールバック / ロガー ---
    callbacks = []
    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.name}-task{args.task_idx}",
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            reinit=True,
        )
        if args.task_idx == 0:
            wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    else:
        wandb_logger = None

    if args.save_checkpoint:
        ckpt = Checkpointer(
            args,
            logdir=args.checkpoint_dir,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    if args.auto_umap:
        assert _umap_available, "UMAP is not currently available; install with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,
        )
        callbacks.append(auto_umap)

    # --- Trainer 構築 ---
    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
    )

    model.current_task_idx = args.task_idx

    # --- 学習 ---
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loaders, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
