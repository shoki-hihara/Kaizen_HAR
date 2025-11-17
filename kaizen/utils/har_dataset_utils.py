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
    
    タスクごとに：
        train_task{i}, test_task{i} の DataLoader を返す。
    """

    # --------------------------
    # 1) train/test データ読み込み
    # --------------------------
    train_dataset = load_wisdm_dataset(data_dir, split="train")
    test_dataset  = load_wisdm_dataset(data_dir, split="test")

    # --------------------------
    # 2) タスク i のクラスリスト（必ず int にそろえる）
    # --------------------------
    raw_task_classes = tasks[task_idx]          # 例: [17, 2, 0] or [tensor(17), ...]
    task_classes = set(int(c) for c in raw_task_classes)

    # デバッグログ
    unique_train = sorted({int(y) for y in train_dataset.targets})
    unique_test  = sorted({int(y) for y in test_dataset.targets})

    print(f"[DEBUG][HAR] Task {task_idx} classes: {sorted(task_classes)}")
    print(f"[DEBUG][HAR] Unique train targets: {unique_train}")
    print(f"[DEBUG][HAR] Unique test targets : {unique_test}")

    # --------------------------
    # 3) タスク i に対応するサブセット生成
    # --------------------------
    train_indices = [
        i for i, y in enumerate(train_dataset.targets)
        if int(y) in task_classes
    ]
    test_indices = [
        i for i, y in enumerate(test_dataset.targets)
        if int(y) in task_classes
    ]

    print(f"[DEBUG][HAR] #train indices for task {task_idx}: {len(train_indices)}")
    print(f"[DEBUG][HAR] #test indices  for task {task_idx}: {len(test_indices)}")

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            f"[HAR ERROR] No samples found for task {task_idx} with classes {sorted(task_classes)}. "
            f"train unique targets={unique_train}, test unique targets={unique_test}. "
            "クラスIDの対応（0 始まり/1 始まりなど）を確認してください。"
        )

    train_subset = Subset(train_dataset, train_indices)
    test_subset  = Subset(test_dataset, test_indices)

    # --------------------------
    # 4) DataLoader を返却
    # --------------------------
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return {f"task{task_idx}": train_loader}, {f"task{task_idx}": test_loader}
