import torch
import torch.nn as nn

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, n_features=1):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # In DLinear, we decompose into trend and seasonal parts
        self.decomposer = SeriesDecomp(kernel_size=25)
        
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        
    def forward_hidden(self, x):
        """
        Extract hidden representation (the 'first half' or relevant embeddings)
        x: (B, L, D)
        """
        # (B, L, D) -> (B, D, L)
        x_transposed = x.transpose(1, 2)
        seasonal, trend = self.decomposer(x_transposed)
        # Hidden could be the concatenation of seasonal and trend
        hidden = torch.cat([seasonal, trend], dim=-1) # (B, D, 2L)
        return hidden

    def forward(self, x):
        """
        Complete prediction
        """
        x_transposed = x.transpose(1, 2)
        seasonal, trend = self.decomposer(x_transposed)
        
        res_seasonal = self.linear_seasonal(seasonal)
        res_trend = self.linear_trend(trend)
        
        res = res_seasonal + res_trend
        return res.transpose(1, 2) # (B, P, D)

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding to keep the same length
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, self.kernel_size // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x
