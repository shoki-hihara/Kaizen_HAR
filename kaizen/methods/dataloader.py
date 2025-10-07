# my_har_dataloader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WISDMDataset(Dataset):
    def __init__(self, data_path, split="train"):
        """
        data_path: ディレクトリ。中に train_X.npy, train_y.npy, test_X.npy, test_y.npy がある
        split: "train" or "test"
        """
        if split == "train":
            self.X = np.load(os.path.join(data_path, "train_X.npy"))
            self.y = np.load(os.path.join(data_path, "train_y.npy"))
        else:
            self.X = np.load(os.path.join(data_path, "test_X.npy"))
            self.y = np.load(os.path.join(data_path, "test_y.npy"))

        # torch tensor に変換
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X の shape は (channel, timesteps) を想定
        return self.X[idx], self.y[idx]


def load_wisdm_dataset(data_dir, split="train"):
    return WISDMDataset(data_dir, split=split)


def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
