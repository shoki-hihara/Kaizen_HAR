# main_pretrain.py (HAR対応 + 蒸留維持, 完全版)
# 大元の構成を極力そのままに保ちつつ、wisdm2019 を含む実験に対応

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
        assert args.num_classes % args.num_tasks == 0
        torch.manual_seed(args.split_seed)
        tasks = torch.randperm(args.num_classes).chunk(args.num_tasks)
        print("Task Classes:", tasks)

    # 乱数固定
    seed_everything(args.global_seed)

    # --- データ準備（大元ロジックを踏襲） ---
    if not args.dali:
        # 変換の構築（unique_augs>1 なら非対称Aug）
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

            # ✅ wisdm2019 を大元の分岐に含める（既に大元も記載あり）
            if args.dataset in ["cifar10", "cifar100", "wisdm2019"]:
                size_crops = [32, 24]
            elif args.dataset == "stl10":
                size_crops = [96, 58]
            else:  # imagenet or custom
                size_crops = [224, 96]

            transform = prepare_multicrop_transform(
                transform,
                size_crops=size_crops,
                num_crops=[args.num_crops, args.num_small_crops],
            )
        else:
            if args.num_crops != 2:
                assert args.method == "wmse"

            # Online Eval 用の transform は最後のもの
            online_eval_transform = transform[-1] if isinstance(transform, list) else transform
            task_transform = prepare_n_crop_transform(transform, num_crops=args.num_crops)

        # 学習/評価用データセット
        train_dataset, online_eval_dataset = prepare_datasets(
            args.dataset,
            task_transform=task_transform,
            online_eval_transform=online_eval_transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            no_labels=args.no_labels,
        )

        # タスク分割（class-split）
        task_dataset, tasks = split_dataset(
            train_dataset,
            tasks=tasks,
            task_idx=args.task_idx,
            num_tasks=args.num_tasks,
            split_strategy=args.split_strategy,
            split_seed=args.split_seed,
        )

        # 現タスクの loader
        task_loader = prepare_dataloader(
            task_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        train_loaders = {f"task{args.task_idx}": task_loader}

        # ✅ リプレイ（見たことあるタスクからのサブセット）
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

        # ✅ Online eval の学習データ供給元（all/current/seen）
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
                # 大元の「最大サイズ追従」ポリシーを維持
                batch_size=-(-len(online_eval_dataset_final) // len(task_loader)),
                num_workers=args.num_workers,
            )
            train_loaders.update({"online_eval": online_eval_loader})

    # --- 検証用の分類データローダ（ある場合のみ） ---
    if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
        val_loader = None
    elif args.dataset in ["imagenet100", "imagenet"] and args.val_dir is None:
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

    # --- 手法クラスの構築（蒸留ラッパはここで差し込む：大元の肝） ---
    assert args.method in METHODS, f"Choose from {METHODS.keys()}"
    MethodClass = METHODS[args.method]

    if args.dali:
        assert _dali_avaliable, "Dali is not currently available; install with [dali]."
        MethodClass = types.new_class(f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass))

    # ✅ 蒸留の有無はここで決定（大元準拠）
    if args.distiller_library == "default":
        if args.distiller:
            MethodClass = DISTILLERS[args.distiller](MethodClass)
    elif args.distiller_library == "factory":
        if args.distiller:
            # 凍結モデル（前タスク）の自動取込をサポートする基底ラッパ
            MethodClass = base_frozen_model_factory(MethodClass)
            # 特徴蒸留：current.z ↔ frozen.frozen_z
            MethodClass = DISTILLER_FACTORIES[args.distiller](
                MethodClass,
                distill_current_key="z",
                distill_frozen_key="frozen_z",
                output_dim=args.output_dim,
                class_tag="feature_extractor",
            )
            # 分類器蒸留（任意）：current.logits ↔ frozen.frozen_logits
            if args.classifier_training and args.distiller_classifier is not None:
                MethodClass = DISTILLER_FACTORIES[args.distiller_classifier](
                    MethodClass,
                    distill_current_key="classifier_logits",
                    distill_frozen_key="frozen_logits",
                    output_dim=args.num_classes,
                    class_tag="classifier",
                )

    # モデル生成（class-split 時は tasks を渡す）
    model = MethodClass(**args.__dict__, tasks=tasks if args.split_strategy == "class" else None)

    # --- 事前学習モデル/再開 ---
    # どちらか一方のみ許可
    assert [args.resume_from_checkpoint, args.pretrained_model].count(True) <= 1

    if args.resume_from_checkpoint:
        pass  # Trainer 側に委譲（大元準拠）
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
        # checkpoint_callback=False,  # PL 1.9-系では非推奨
        # terminate_on_nan=True,
    )

    # 現在タスクインデックス（メソッド側で参照）
    model.current_task_idx = args.task_idx

    # --- 学習 ---
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loaders, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
