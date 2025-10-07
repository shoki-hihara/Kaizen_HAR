import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_wisdm_dataset(data_dir: str, split: str = "train"):
    """
    前処理済み WISDM2019 データセットを読み込む

    Args:
        data_dir (str): データセット格納ディレクトリ
        split (str): "train" or "test"

    Returns:
        dataset (TensorDataset): TensorDataset (X, y)
    """
    X_path = f"{data_dir}/{split}_X.npy"
    y_path = f"{data_dir}/{split}_y.npy"

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    # HAR用に (batch, channel, timestep) 形状に変換
    if X.ndim == 3:  # (batch, timestep, channel)
        X = X.transpose(0, 2, 1)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return dataset

def get_dataloader(data_dir: str, split: str = "train", batch_size: int = 64, shuffle: bool = True, num_workers: int = 2):
    dataset = load_wisdm_dataset(data_dir, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
