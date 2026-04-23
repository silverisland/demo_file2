import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from data_provider.fusion_dataset import UnifiedDataset, FusionFeatureDataset, collate_fn, fusion_collate_fn
from models.fusion import FusionFeatureModel

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
        x_expert = torch.stack([batch_dict[col] for col in self.target_cols], dim=-1) # (B, L, C)
            
        # 2. Model forward pass (Projection + Transformer layer)
        hidden = self.projection(x_expert) # (B, L, d_model)
        hidden = self.transformer(hidden) # (B, L, d_model)
        return hidden 

    def forward(self, batch_dict):
        return self.forward_hidden(batch_dict)

def extract_and_save_features(model, dataloader, save_path, device='cpu'):
    """
    Extract hidden features and save to a .npy file.
    """
    model.to(device)
    model.eval()
    all_hiddens = []
    
    print(f"Extracting features to {save_path}...")
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
            hidden = model.forward_hidden(batch_device)
            all_hiddens.append(hidden.cpu().numpy())
    
    final_features = np.concatenate(all_hiddens, axis=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, final_features)
    print(f"Saved {final_features.shape} features.")
    return final_features.shape

def run_demo():
    # 1. Create dummy data
    num_samples = 100
    seq_len = 672
    pred_len = 192
    data = {
        'timestamp_win': pd.date_range(start='2023-01-01', periods=num_samples, freq='H'),
        'observe_power': [np.random.randn(seq_len) for _ in range(num_samples)],
        'observe_power_future': [np.random.randn(pred_len) for _ in range(num_samples)],
        'GHI_solargis': [np.random.randn(seq_len) for _ in range(num_samples)],
        'GHI_solargis_future': [np.random.randn(pred_len) for _ in range(num_samples)],
    }
    df = pd.DataFrame(data)
    
    # 2. Initialize Row-based Unified Dataset for extraction
    dataset = UnifiedDataset(df)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)
    
    # 3. Phase 1: Feature Extraction
    expert_a = ExpertModelDemo(target_cols=['observe_power'], d_model=64)
    expert_b = ExpertModelDemo(target_cols=['GHI_solargis'], d_model=128)
    
    feature_paths = {
        'expert_a': 'data/features/expert_a.npy',
        'expert_b': 'data/features/expert_b.npy'
    }
    
    shape_a = extract_and_save_features(expert_a, dataloader, feature_paths['expert_a'])
    shape_b = extract_and_save_features(expert_b, dataloader, feature_paths['expert_b'])
    
    # 4. Phase 2: Fast Fusion Iteration
    print("\n>>> Starting Fast Fusion Iteration Demo...")
    
    # Initialize FusionFeatureDataset (mmap mode)
    fusion_dataset = FusionFeatureDataset(
        df, 
        feature_paths, 
        target_cols=['observe_power_future']
    )
    fusion_loader = DataLoader(
        fusion_dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=fusion_collate_fn
    )
    
    # Initialize FusionFeatureModel
    # Note: expert_dims should match extracted shapes (B, L, D) -> dim is D
    expert_dims = {
        'expert_a': shape_a[-1],
        'expert_b': shape_b[-1]
    }
    
    fusion_model = FusionFeatureModel(
        expert_dims=expert_dims,
        pred_len=pred_len,
        n_features=1, # predicting observe_power_future
        d_fusion=64,
        num_experts=2,
        use_dynamic_queries=True
    )
    
    # 5. Run a few training steps
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    fusion_model.train()
    for i, (feats, targets, x_input) in enumerate(fusion_loader):
        optimizer.zero_grad()
        
        # Forward pass with pre-computed features
        outputs = fusion_model(feats, x_input) # (B, P, n_features)
        
        # Loss calculation
        target_tensor = targets['observe_power_future'].unsqueeze(-1) # (B, P, 1)
        loss = criterion(outputs, target_tensor)
        
        loss.backward()
        optimizer.step()
        
        if i % 2 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")
            print(f"  Output shape: {outputs.shape}")
        if i >= 4: break

    print("\nDemo successful! Hidden vectors saved and fusion model iterated.")

if __name__ == "__main__":
    run_demo()
