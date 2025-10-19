import os
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch

class WISDMDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (np.ndarray): shape (num_samples, window_size, 3)
            labels (np.ndarray): shape (num_samples,)
            transform (callable, optional): optional transform to be applied
                on a sample
        """
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def load_wisdm_csv(file_path):
    """
    Load WISDM CSV and return numpy arrays
    """
    df = pd.read_csv(file_path)
    # Assume columns: user, activity, timestamp, x, y, z
    le = LabelEncoder()
    df['activity'] = le.fit_transform(df['activity'])
    return df, le

def window_data(df, window_size=384, step_size=384):
    """
    Slice data into windows
    """
    X = []
    y = []

    for user_id in df['user'].unique():
        user_data = df[df['user'] == user_id]
        activities = user_data['activity'].values
        sensor_values = user_data[['x','y','z']].values

        start = 0
        while start + window_size <= len(sensor_values):
            X.append(sensor_values[start:start+window_size])
            # Label: most frequent activity in window
            label_window = activities[start:start+window_size]
            counts = np.bincount(label_window)
            y.append(np.argmax(counts))
            start += step_size

    return np.array(X), np.array(y)

def prepare_wisdm_dataloaders(csv_path, batch_size=64, window_size=384, step_size=384, val_ratio=0.2, num_workers=4):
    df, label_encoder = load_wisdm_csv(csv_path)
    X, y = window_data(df, window_size, step_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=42, stratify=y
    )

    train_dataset = WISDMDataset(X_train, y_train)
    val_dataset = WISDMDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, label_encoder
