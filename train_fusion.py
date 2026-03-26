import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import get_dataloaders
from models.dlinear import DLinear
from models.patchtst import PatchTST
from models.itransformer import iTransformer
from models.timesnet import TimesNet
from models.fusion import FusionModel
import os
from tqdm import tqdm
import numpy as np

def evaluate_and_compare(fusion_model, base_models, test_loader, device):
    fusion_model.eval()
    for m in base_models.values():
        m.eval()
        
    criterion = nn.MSELoss()
    fusion_loss = 0
    avg_loss = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Fusion Prediction
            fusion_preds = fusion_model(x)
            fusion_loss += criterion(fusion_preds, y).item()
            
            # Mean Prediction
            base_preds = []
            for name, m in base_models.items():
                base_preds.append(m(x))
            
            avg_preds = torch.stack(base_preds).mean(dim=0)
            avg_loss += criterion(avg_preds, y).item()
            
    print(f"\nFinal Test Performance:")
    print(f"Fusion Model MSE: {fusion_loss/len(test_loader):.6f}")
    print(f"Mean Ensemble MSE: {avg_loss/len(test_loader):.6f}")
    
    if fusion_loss < avg_loss:
        print("Success! Fusion model outperformed the mean ensemble.")
    else:
        print("Fusion model did not outperform the mean ensemble. Try training longer or adjusting the head.")

def train_fusion(fusion_model, train_loader, val_loader, device, epochs=10):
    print("Training Fusion Model Head...")
    criterion = nn.MSELoss()
    # Only optimize fusion_head parameters
    optimizer = optim.Adam(fusion_model.fusion_head.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        fusion_model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = fusion_model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        fusion_model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = fusion_model(x)
                val_loss += criterion(preds, y).item()
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    seq_len = 96
    pred_len = 24
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size, seq_len, pred_len)
    
    # Initialize base models and load weights
    base_models = {
        "DLinear": DLinear(seq_len, pred_len).to(device),
        "PatchTST": PatchTST(seq_len, pred_len).to(device),
        "iTransformer": iTransformer(seq_len, pred_len).to(device),
        "TimesNet": TimesNet(seq_len, pred_len).to(device)
    }
    
    for name, model in base_models.items():
        path = f'checkpoints/{name}.pth'
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded weights for {name}")
        else:
            print(f"Warning: {path} not found. Using randomly initialized weights.")
            
    # Initialize fusion model
    fusion_model = FusionModel(base_models, seq_len, pred_len, device=device).to(device)
    
    train_fusion(fusion_model, train_loader, val_loader, device, epochs=10)
    
    evaluate_and_compare(fusion_model, base_models, test_loader, device)
