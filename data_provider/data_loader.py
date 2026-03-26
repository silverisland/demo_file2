import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], 0, 0 # Add markers if needed, using 0 for now to match interface

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
    
    return x_data, y_data
