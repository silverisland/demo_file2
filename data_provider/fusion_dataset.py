import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class UnifiedDataset(Dataset):
    """
    Row-based dataset for fusion models. 
    Each item is a single row from the dataframe, excluding timestamps.
    Metadata (column names) is preserved for expert models to select their own features.
    """
    def __init__(self, df, timestamp_cols=['date', 'time', 'timestamp', 'timestamp_win']):
        self.df_raw = df.copy()
        
        # 1. Identify and separate timestamp columns
        self.timestamp_cols = [col for col in timestamp_cols if col in self.df_raw.columns]
        self.feature_cols = [col for col in self.df_raw.columns if col not in self.timestamp_cols]
        
        # 2. Store clean column names for expert models to reference
        self.column_names = self.feature_cols
        
    def __len__(self):
        return len(self.df_raw)

    def __getitem__(self, index):
        # Returns a single row as a dictionary
        row = self.df_raw.iloc[index]
        sample = {col: row[col] for col in self.feature_cols}
        sample['index'] = index
        return sample

def collate_fn(batch):
    """
    Collates single rows into a batch.
    Returns a dictionary of tensors where keys are column names.
    """
    feature_keys = [k for k in batch[0].keys() if k != 'index']
    collated = {}
    
    for key in feature_keys:
        data_list = [b[key] for b in batch]
        # Check if it's a list of numpy arrays or scalars
        if isinstance(data_list[0], np.ndarray):
            # Batch shape: (Batch, SeqLen)
            collated[key] = torch.from_numpy(np.stack(data_list)).float()
        else:
            # Batch shape: (Batch,)
            collated[key] = torch.tensor(data_list).float()
            
    collated['index'] = torch.tensor([b['index'] for b in batch])
    collated['column_names'] = feature_keys
    
    return collated
