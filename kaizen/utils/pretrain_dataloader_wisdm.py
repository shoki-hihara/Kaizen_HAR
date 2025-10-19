import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def prepare_wisdm_dataloaders(data_dir, batch_size=64, val_ratio=0.1, num_workers=2):
    """
    WISDM 用 DataLoader 準備（pretrain 用）
    - train データのみ使用
    - train の一部を val に分割
    - npy ファイルは前処理済み（ウインドウ化済み）を想定
    """

    # -------------------------------
    # 1️⃣ npy データ読み込み
    # -------------------------------
    train_X_path = os.path.join(data_dir, "train_X.npy")
    train_y_path = os.path.join(data_dir, "train_y.npy")

    X_train = np.load(train_X_path)
    y_train = np.load(train_y_path)

    # Tensor に変換
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()

    # -------------------------------
    # 2️⃣ train / val 分割
    # -------------------------------
    num_val = int(len(X_train) * val_ratio)
    num_train = len(X_train) - num_val

    train_dataset, val_dataset = random_split(
        TensorDataset(X_train, y_train),
        lengths=[num_train, num_val],
        generator=torch.Generator().manual_seed(42)  # 再現性
    )

    # -------------------------------
    # 3️⃣ DataLoader 作成
    # -------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
