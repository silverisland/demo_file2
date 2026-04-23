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

        # 3. Also pre-process timestamp columns
        for col in self.timestamp_cols:
            vals = df[col].values
            if len(vals) > 0 and isinstance(vals[0], np.ndarray):
                self.data_dict[col] = np.stack(vals)
            else:
                self.data_dict[col] = vals

        self.column_names = self.feature_cols
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Fast indexing from dictionary of pre-stacked arrays (features + timestamps)
        sample = {col: self.data_dict[col][index] for col in self.feature_cols + self.timestamp_cols}
        sample['index'] = index
        return sample

class FusionFeatureDataset(Dataset):
    """
    Dataset for fast fusion model iteration using pre-computed expert hidden states.
    Uses memory-mapping (mmap_mode='r') to avoid loading all features into RAM.
    """
    def __init__(self, df, feature_paths, target_cols):
        self.df = df
        self.target_cols = target_cols
        self.feature_paths = feature_paths
        
        # Open memory-mapped files
        self.experts_data = {
            name: np.load(path, mmap_mode='r') 
            for name, path in feature_paths.items()
        }
        
        # Verify alignment
        for name, data in self.experts_data.items():
            if len(data) != len(df):
                raise ValueError(f"Expert feature '{name}' has {len(data)} samples, but DataFrame has {len(df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 1. Load expert features for this index
        # Returns dict: {'expert_a': torch.Tensor(L, D1), ...}
        feats = {
            name: torch.from_numpy(data[index].copy()).float()
            for name, data in self.experts_data.items()
        }
        
        # 2. Load target columns from DataFrame
        targets = {}
        for col in self.target_cols:
            val = self.df.iloc[index][col]
            if isinstance(val, np.ndarray):
                targets[col] = torch.from_numpy(val).float()
            else:
                targets[col] = torch.tensor(val).float()
        
        # 3. Load additional input 'x' if present in df (for query generation)
        input_x = None
        if 'observe_power' in self.df.columns:
             input_x = torch.from_numpy(self.df.iloc[index]['observe_power']).float()
        
        return feats, targets, input_x

def fusion_collate_fn(batch):
    """
    Collate function for FusionFeatureDataset.
    Returns (collated_feats, collated_targets, collated_x)
    """
    expert_names = batch[0][0].keys()
    target_names = batch[0][1].keys()
    
    collated_feats = {
        name: torch.stack([sample[0][name] for sample in batch])
        for name in expert_names
    }
    
    collated_targets = {
        name: torch.stack([sample[1][name] for sample in batch])
        for name in target_names
    }
    
    collated_x = None
    if batch[0][2] is not None:
        collated_x = torch.stack([sample[2] for sample in batch])
        if collated_x.dim() == 2:
            collated_x = collated_x.unsqueeze(-1)
            
    return collated_feats, collated_targets, collated_x

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
