import torch
from torch.utils.data import DataLoader, Subset
from kaizen.methods.dataloader import load_wisdm_dataset

def prepare_task_datasets(
    data_dir,
    task_idx,
    tasks,
    batch_size=256,
    num_workers=2,
    replay=False,
    replay_proportion=0.01,
):
    """
    Kaizen の「class incremental」に合わせて
    WISDM データを task ごとに分割してロードするためのユーティリティ。

    ※ 論文再現のため、全タスク分の DataLoader を返す。
       main_linear 側で task_idx に対応する train/val と
       0〜task_idx までの past_task_loaders を選ぶ。
    """

    # --------------------------
    # 1) train/test データ読み込み
    # --------------------------
    train_dataset = load_wisdm_dataset(data_dir, split="train")
    test_dataset  = load_wisdm_dataset(data_dir, split="test")

    unique_train = sorted({int(y) for y in train_dataset.targets})
    unique_test  = sorted({int(y) for y in test_dataset.targets})

    # ここで全タスク分の DataLoader を作る
    train_loaders = {}
    test_loaders  = {}

    for t, raw_task_classes in enumerate(tasks):
        # --------------------------
        # 2) タスク t のクラスリスト
        # --------------------------
        task_classes = set(int(c) for c in raw_task_classes)

        print(f"[DEBUG][HAR] Task {t} classes: {sorted(task_classes)}")
        print(f"[DEBUG][HAR] Unique train targets: {unique_train}")
        print(f"[DEBUG][HAR] Unique test targets : {unique_test}")

        # --------------------------
        # 3) タスク t に対応するサブセット生成
        # --------------------------
        train_indices = [
            i for i, y in enumerate(train_dataset.targets)
            if int(y) in task_classes
        ]
        test_indices = [
            i for i, y in enumerate(test_dataset.targets)
            if int(y) in task_classes
        ]

        print(f"[DEBUG][HAR] #train indices for task {t}: {len(train_indices)}")
        print(f"[DEBUG][HAR] #test indices  for task {t}: {len(test_indices)}")

        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError(
                f"[HAR ERROR] No samples found for task {t} with classes {sorted(task_classes)}. "
                f"train unique targets={unique_train}, test unique targets={unique_test}. "
                "クラスIDの対応（0 始まり/1 始まりなど）を確認してください。"
            )

        train_subset = Subset(train_dataset, train_indices)
        test_subset  = Subset(test_dataset,  test_indices)

        # --------------------------
        # 4) DataLoader を dict に格納
        # --------------------------
        train_loaders[f"task{t}"] = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )

        test_loaders[f"task{t}"] = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    if replay:
        # 論文再現の最初のステップとして、ここではリプレイは未対応にしておく。
        # 必要になったら、pretrain 側と同じ設計で追加していく想定。
        print("[WARN][HAR] replay=True ですが、prepare_task_datasets では未対応のため無視します。")

    return train_loaders, test_loaders
