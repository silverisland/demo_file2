import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class UnifiedDataset(Dataset):
    """
    A unified dataset for fusion models that provides raw data and column metadata.
    This allows expert models to perform their own internal preprocessing (e.g., column selection, scaling).
    """
    def __init__(self, df, seq_len=96, pred_len=24, target_col='OT', time_features=True):
        self.df_raw = df.copy()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.time_features = time_features
        
        # Get column names for the expert models to reference
        self.column_names = list(self.df_raw.columns)
        
        # Pre-process time features if requested
        if 'date' in self.df_raw.columns:
            self.df_raw['date'] = pd.to_datetime(self.df_raw['date'])
            if self.time_features:
                self.df_raw['month'] = self.df_raw.date.dt.month
                self.df_raw['day'] = self.df_raw.date.dt.day
                self.df_raw['weekday'] = self.df_raw.date.dt.weekday
                self.df_raw['hour'] = self.df_raw.date.dt.hour
                self.time_cols = ['month', 'day', 'weekday', 'hour']
            else:
                self.time_cols = []
        else:
            self.time_cols = []

        # Values as float32 tensor
        # We don't scale here because expert models might need raw values or their own scalers
        self.data_values = self.df_raw.drop(columns=['date'] if 'date' in self.df_raw.columns else []).values.astype(np.float32)
        
    def __len__(self):
        return len(self.data_values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_values[s_begin:s_end]
        seq_y = self.data_values[r_begin:r_end]

        # Return as a dictionary so it can be easily passed to models
        return {
            'x_raw': seq_x,
            'y_raw': seq_y,
            'column_names': self.column_names,
            'index': index
        }

def collate_fn(batch):
    """
    Custom collate function to handle the dictionary output of UnifiedDataset.
    """
    x_raw = torch.from_numpy(np.stack([b['x_raw'] for b in batch])).float()
    y_raw = torch.from_numpy(np.stack([b['y_raw'] for b in batch])).float()
    column_names = batch[0]['column_names']
    
    return {
        'x_raw': x_raw,
        'y_raw': y_raw,
        'column_names': column_names
    }
