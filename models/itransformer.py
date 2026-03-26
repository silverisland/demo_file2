import torch
import torch.nn as nn

class iTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=128, n_heads=4, n_layers=2):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Inverted Transformer: each variate is a token, time points are features
        self.enc_embedding = nn.Linear(seq_len, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model, pred_len)

    def forward_hidden(self, x):
        """
        Extract hidden representation
        x: (B, L, D)
        """
        # (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        # (B, D, L) -> (B, D, d_model)
        hidden = self.enc_embedding(x)
        hidden = self.transformer_encoder(hidden)
        return hidden # (B, D, d_model)

    def forward(self, x):
        hidden = self.forward_hidden(x)
        # (B, D, d_model) -> (B, D, P)
        out = self.head(hidden)
        return out.transpose(1, 2) # (B, P, D)
