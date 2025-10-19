# my_har_dataloader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WISDMDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        x_path = f"{data_dir}/{split}_X.npy"
        y_path = f"{data_dir}/{split}_y.npy"

        # データ存在確認
        assert os.path.exists(x_path), f"Missing {x_path}"
        assert os.path.exists(y_path), f"Missing {y_path}"

        # np.load + デバッグ出力
        self.X = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)

        print(f"[WISDMDataset] Loaded {split} set: X={self.X.shape}, y={self.y.shape}")

        # torch.tensor化
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def load_wisdm_dataset(data_dir, split="train"):
    return WISDMDataset(data_dir, split=split)


def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
