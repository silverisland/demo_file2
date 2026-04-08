import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data_provider.fusion_dataset import UnifiedDataset, collate_fn

# Example: How an expert model should be implemented or wrapped
class ExpertModelDemo(nn.Module):
    def __init__(self, target_cols=['observe_power', 'GHI_solargis'], d_model=128):
        super(ExpertModelDemo, self).__init__()
        self.target_cols = target_cols
        self.projection = nn.Linear(len(target_cols), d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward_hidden(self, batch_dict):
        """
        Modified for dictionary-based batch format
        Each selected feature is (B, L)
        Returns (B, L, D_model)
        """
        # 1. Expert-specific column selection: Stack selected (B, L) features into (B, L, C)
        # Assuming all target_cols have the same sequence length
        x_expert = torch.stack([batch_dict[col] for col in self.target_cols], dim=-1) # (B, L, C)
            
        # 2. Model forward pass (Projection + Transformer layer)
        hidden = self.projection(x_expert) # (B, L, d_model)
        hidden = self.transformer(hidden) # (B, L, d_model)
        return hidden 

    def forward(self, batch_dict):
        return self.forward_hidden(batch_dict)

def run_demo():
    # 1. Create dummy data as described in dataset_description.md
    num_samples = 100
    data = {
        'timestamp_win': pd.date_range(start='2023-01-01', periods=num_samples, freq='H'),
        'observe_power': [np.random.randn(672) for _ in range(num_samples)],
        'observe_power_future': [np.random.randn(192) for _ in range(num_samples)],
        'GHI_solargis': [np.random.randn(672) for _ in range(num_samples)],
        'GHI_solargis_future': [np.random.randn(192) for _ in range(num_samples)],
        'TEMP_solargis': [np.random.randn(672) for _ in range(num_samples)],
        'TEMP_solargis_future': [np.random.randn(192) for _ in range(num_samples)],
    }
    df = pd.DataFrame(data)
    
    # 2. Initialize Row-based Unified Dataset
    dataset = UnifiedDataset(df)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # 3. Initialize Expert Model
    # Example expert focusing on history data
    expert_model = ExpertModelDemo(target_cols=['observe_power', 'GHI_solargis'])
    
    # 4. Loop Through Batches
    print(f"Total rows: {len(dataset)}")
    print(f"Total batches: {len(dataloader)}")
    print(f"Feature columns: {dataset.column_names}")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            hidden_output = expert_model.forward_hidden(batch)
            
            if i % 5 == 0:
                print(f"Batch {i}:")
                # Showing shapes of some selected columns
                print(f"  'observe_power' shape: {batch['observe_power'].shape}")
                print(f"  'GHI_solargis' shape: {batch['GHI_solargis'].shape}")
                print(f"  Expert Hidden Output Shape: {hidden_output.shape}")
        
    print("\nDemo successful with unified row-based data!")

if __name__ == "__main__":
    run_demo()
