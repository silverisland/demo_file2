import torch
import torch.nn as nn

class PatchTST(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len=16, stride=8, d_model=128, n_heads=4, n_layers=2):
        super(PatchTST, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        
        self.W_P = nn.Linear(patch_len, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model * self.num_patch, pred_len)

    def forward_hidden(self, x):
        """
        Extract hidden representation
        x: (B, L, D)
        """
        # (B, L, D) -> (B, L) if D=1
        x = x.squeeze(-1)
        # Patching (B, L) -> (B, num_patch, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # (B, num_patch, patch_len) -> (B, num_patch, d_model)
        hidden = self.W_P(patches)
        hidden = self.transformer_encoder(hidden)
        return hidden # (B, num_patch, d_model)

    def forward(self, x):
        hidden = self.forward_hidden(x)
        hidden = hidden.flatten(start_dim=1)
        out = self.head(hidden)
        return out.unsqueeze(-1) # (B, P, D)
