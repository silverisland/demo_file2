import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def generate_dummy_data(n_samples=1000, seq_len=96, pred_len=24, n_features=1):
    """
    Generate synthetic time series data: sine waves + trend + noise
    """
    t = np.linspace(0, 100 * np.pi, n_samples + seq_len + pred_len)
    # Sine wave + Trend + Noise
    data = np.sin(t) + 0.1 * t + np.random.normal(0, 0.1, len(t))
    data = data.reshape(-1, n_features)
    
    x_data = []
    y_data = []
    
    for i in range(len(data) - seq_len - pred_len):
        x_data.append(data[i : i + seq_len])
        y_data.append(data[i + seq_len : i + seq_len + pred_len])
        
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Split: Train 70%, Val 10%, Test 20%
    n_train = int(len(x_data) * 0.7)
    n_val = int(len(x_data) * 0.1)
    
    train_x, train_y = x_data[:n_train], y_data[:n_train]
    val_x, val_y = x_data[n_train : n_train + n_val], y_data[n_train : n_train + n_val]
    test_x, test_y = x_data[n_train + n_val :], y_data[n_train + n_val :]
    
    return train_x, train_y, val_x, val_y, test_x, test_y

def get_dataloaders(batch_size=32, seq_len=96, pred_len=24, n_features=1):
    train_x, train_y, val_x, val_y, test_x, test_y = generate_dummy_data(
        n_samples=2000, seq_len=seq_len, pred_len=pred_len, n_features=n_features
    )
    
    train_set = TimeSeriesDataset(train_x, train_y)
    val_set = TimeSeriesDataset(val_x, val_y)
    test_set = TimeSeriesDataset(test_x, test_y)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    for x, y in train_loader:
        print(f"Batch shape x: {x.shape}, y: {y.shape}")
        break
