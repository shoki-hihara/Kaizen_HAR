import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from kaizen.utils.har_transforms import Random3DRotation, RandomScaling, TimeWarp

class WISDMDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        x_path = f"{data_dir}/{split}_X.npy"
        y_path = f"{data_dir}/{split}_y.npy"

        assert os.path.exists(x_path), f"Missing {x_path}"
        assert os.path.exists(y_path), f"Missing {y_path}"

        self.X = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)

        print(f"[WISDMDataset] Loaded {split} set: X={self.X.shape}, y={self.y.shape}")

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        # 🔸 Kaizen の split_dataset / その他が期待する属性
        self.num_classes = 18                # WISDM2019 のクラス数
        self.classes = list(range(18))       # 中身は index でも OK
        self.targets = self.y                # CIFAR と同じ名前に合わせる

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def build_har_train_transform():
    """WISDM 用の学習時データ拡張（TPN の 3 変換）"""
    rot = Random3DRotation(max_angle=np.pi / 6)
    scale = RandomScaling(scale_range=(0.9, 1.1))
    warp = TimeWarp(sigma=0.2, knot=4)

    def _transform(x):
        # x: Tensor [T, C] など
        x = scale(x)
        x = rot(x)
        x = warp(x)
        return x

    return _transform


def load_wisdm_dataset(data_dir, split="train", transform=None):
    """
    WISDMDataset のシンプルなラッパ。
    pretrain でも linear でも同じインターフェースで使える。

    Args:
        data_dir (str): /.../wisdm2019_npy
        split (str): "train" / "val" / "test"
        transform: データ拡張（pretrainのみ使用）
    """
    return WISDMDataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
    )


def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(shuffle is True),  # train では drop_last=True にすることが多い
    )
