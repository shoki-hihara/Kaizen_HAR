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
    # 2) タスク i のクラスリスト
    # --------------------------
    task_classes = tasks[task_idx]  # e.g., [17, 2, 0]
    task_classes = set(task_classes)

    # --------------------------
    # 3) タスク i に対応するサブセット生成
    # --------------------------
    train_indices = [i for i, y in enumerate(train_dataset.targets) if y.item() in task_classes]
    test_indices  = [i for i, y in enumerate(test_dataset.targets)  if y.item() in task_classes]

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
        drop_last=False
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return {f"task{task_idx}": train_loader}, {f"task{task_idx}": test_loader}
