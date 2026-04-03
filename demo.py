import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data_provider.fusion_dataset import UnifiedDataset, collate_fn

# Example: How an expert model should be implemented or wrapped
class ExpertModelDemo(nn.Module):
    def __init__(self, target_cols=['OT', 'HUFL'], d_model=128):
        super(ExpertModelDemo, self).__init__()
        self.target_cols = target_cols
        # Expert model's own logic (e.g., its own scaler or parameters)
        self.scaler = None # Could be pre-loaded
        self.projection = nn.Linear(len(target_cols), d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward_hidden(self, batch_dict):
        """
        Requirements for Colleagues:
        1. Your model MUST implement forward_hidden(batch_dict)
        2. batch_dict contains 'x_raw' (B, L, D) and 'column_names' (List of D names)
        3. You must handle your own column selection and scaling here.
        """
        x_raw = batch_dict['x_raw'] # (B, L, D)
        column_names = batch_dict['column_names']
        
        # 1. Expert-specific column selection using column names
        # Some colleagues might use index, some use column titles
        indices = [column_names.index(col) for col in self.target_cols if col in column_names]
        if not indices:
            # Fallback or error handling
            x_expert = x_raw[:, :, :len(self.target_cols)] 
        else:
            x_expert = x_raw[:, :, indices] # (B, L, selected_D)
            
        # 2. Expert-specific preprocessing (if any)
        # x_expert = (x_expert - self.mean) / self.std 
        
        # 3. Model forward pass to extract hidden representation
        hidden = self.projection(x_expert)
        hidden = self.transformer(hidden)
        
        # Return hidden representation (e.g., (B, L, d_model) or (B, d_model))
        # The fusion model will flatten or pool this for the final head
        return hidden 

    def forward(self, batch_dict):
        hidden = self.forward_hidden(batch_dict)
        # Regular forward pass if used standalone
        return hidden.mean(dim=1) # Example output

def run_demo():
    # 1. Create dummy data
    data = {
        'date': pd.date_range(start='2023-01-01', periods=200, freq='H'),
        'OT': np.random.randn(200),
        'HUFL': np.random.randn(200),
        'HULL': np.random.randn(200),
        'MUFL': np.random.randn(200)
    }
    df = pd.DataFrame(data)
    
    # 2. Initialize Unified Dataset
    # This dataset contains all raw information
    dataset = UnifiedDataset(df, seq_len=96, pred_len=24)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # 3. Initialize Expert Model
    # Colleague's model is initialized with their specific needs
    expert_model = ExpertModelDemo(target_cols=['OT', 'HUFL'])
    
    # 4. Loop Through Batches
    print(f"Total batches: {len(dataloader)}")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Simulate Fusion Model calling the expert model
            hidden_output = expert_model.forward_hidden(batch)
            
            if i % 5 == 0: # Print every 5 batches to avoid clutter
                print(f"Batch {i}:")
                print(f"  Batch Keys: {list(batch.keys())}")
                print(f"  x_raw shape: {batch['x_raw'].shape}")
                print(f"  Expert Hidden Output Shape: {hidden_output.shape}")
        
    print("\nDemo successful! All batches processed.")

if __name__ == "__main__":
    run_demo()
