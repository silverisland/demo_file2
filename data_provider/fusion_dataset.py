import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class UnifiedDataset(Dataset):
    """
    Optimized Row-based dataset for fusion models. 
    Pre-converts DataFrame columns to NumPy arrays for fast indexing.
    """
    def __init__(self, df, timestamp_cols=['date', 'time', 'timestamp', 'timestamp_win']):
        # 1. Identify and separate timestamp columns
        self.timestamp_cols = [col for col in timestamp_cols if col in df.columns]
        self.feature_cols = [col for col in df.columns if col not in self.timestamp_cols]
        
        # 2. Pre-process features into a dictionary of arrays
        self.data_dict = {}
        for col in self.feature_cols:
            vals = df[col].values
            # If the column contains numpy arrays (e.g. sequence data), stack them into a single (N, L) array
            if len(vals) > 0 and isinstance(vals[0], np.ndarray):
                self.data_dict[col] = np.stack(vals).astype(np.float32)
            else:
                self.data_dict[col] = vals.astype(np.float32)
        
        self.column_names = self.feature_cols
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Fast indexing from dictionary of pre-stacked arrays
        sample = {col: self.data_dict[col][index] for col in self.feature_cols}
        sample['index'] = index
        return sample

def collate_fn(batch):
    """
    Optimized collate function.
    """
    feature_keys = [k for k in batch[0].keys() if k != 'index']
    collated = {}
    
    # Batching remains the same but inputs are now faster to access
    for key in feature_keys:
        # Using np.array for speed if not already numpy
        data_list = [b[key] for b in batch]
        collated[key] = torch.from_numpy(np.stack(data_list)).float()
            
    collated['index'] = torch.tensor([b['index'] for b in batch])
    collated['column_names'] = feature_keys
    
    return collated
