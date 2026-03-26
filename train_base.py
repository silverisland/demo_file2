import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import get_dataloaders
from models.dlinear import DLinear
from models.patchtst import PatchTST
from models.itransformer import iTransformer
from models.timesnet import TimesNet
import os
from tqdm import tqdm

def train_one_model(model_name, model, train_loader, val_loader, device, epochs=10):
    print(f"Training {model_name}...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_loss += criterion(preds, y).item()
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/{model_name}.pth')
    print(f"Saved {model_name} to checkpoints/{model_name}.pth")

if __name__ == "__main__":
    seq_len = 96
    pred_len = 24
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size, seq_len, pred_len)
    
    models = {
        "DLinear": DLinear(seq_len, pred_len),
        "PatchTST": PatchTST(seq_len, pred_len),
        "iTransformer": iTransformer(seq_len, pred_len),
        "TimesNet": TimesNet(seq_len, pred_len)
    }
    
    for name, model in models.items():
        model.to(device)
        train_one_model(name, model, train_loader, val_loader, device, epochs=5)
