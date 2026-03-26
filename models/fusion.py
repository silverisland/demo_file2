import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, models_dict, seq_len, pred_len, n_features=1, device='cpu'):
        super(FusionModel, self).__init__()
        self.models_dict = nn.ModuleDict(models_dict)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.device = device
        
        # Freeze base models
        for name, model in self.models_dict.items():
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                
        # Calculate concatenation size by passing dummy input
        dummy_x = torch.zeros(1, seq_len, n_features).to(device)
        concat_dim = self._get_concat_dim(dummy_x)
        
        # Initialize fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.pred_len * self.n_features)
        ).to(device)
        
    def _get_concat_dim(self, x):
        hiddens = []
        with torch.no_grad():
            for name, model in self.models_dict.items():
                h = model.forward_hidden(x)
                hiddens.append(h.flatten(start_dim=1))
            concat_hidden = torch.cat(hiddens, dim=-1)
            return concat_hidden.shape[-1]

    def forward(self, x):
        hiddens = []
        # No grad for base models
        with torch.no_grad():
            for name, model in self.models_dict.items():
                h = model.forward_hidden(x)
                hiddens.append(h.flatten(start_dim=1))
        
        concat_hidden = torch.cat(hiddens, dim=-1)
        out = self.fusion_head(concat_hidden)
        return out.view(-1, self.pred_len, self.n_features) # (B, P, D)
