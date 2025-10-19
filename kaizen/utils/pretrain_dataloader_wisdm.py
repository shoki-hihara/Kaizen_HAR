import os
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch

def prepare_wisdm_dataloaders(train_file, test_file, batch_size=64, num_workers=2):
    """
    WISDM2019 の npy 前処理済みデータを読み込み DataLoader を返す
    Args:
        train_file (str): train ファイルのパス (例: "/content/drive/.../train")
        test_file  (str): test ファイルのパス
        batch_size (int)
        num_workers (int)
    Returns:
        train_loader, val_loader
    """
    # npy の読み込み
    X_train = np.load(train_file + "_X.npy")  # shape: (num_samples, window_size, 3)
    y_train = np.load(train_file + "_y.npy")  # shape: (num_samples,)
    X_test  = np.load(test_file  + "_X.npy")
    y_test  = np.load(test_file  + "_y.npy")

    # TensorDataset に変換
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                  torch.from_numpy(y_train).long())
    val_dataset   = TensorDataset(torch.from_numpy(X_test).float(),
                                  torch.from_numpy(y_test).long())

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def prepare_wisdm_dataloaders(train_file, test_file, batch_size=64, num_workers=2):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # train / test は npy で保存されていると仮定
    X_train = np.load(DATA_DIR + "train_X.npy")
    y_train = np.load(DATA_DIR + "train_y.npy")
    X_test = np.load(DATA_DIR + "test_X.npy")
    y_test = np.load(DATA_DIR + "test_y.npy")

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset  = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
