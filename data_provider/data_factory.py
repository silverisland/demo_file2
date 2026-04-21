from torch.utils.data import DataLoader
from data_provider.data_loader import generate_dummy_data
from data_provider.fusion_dataset import UnifiedDataset, collate_fn
import pandas as pd

def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    elif flag == 'val':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    x_data, y_data = generate_dummy_data(
        n_samples=2000, 
        seq_len=args.seq_len, 
        pred_len=args.pred_len, 
        n_features=args.enc_in
    )
    
    # Split: Train 70%, Val 10%, Test 20%
    n_train = int(len(x_data) * 0.7)
    n_val = int(len(x_data) * 0.1)
    
    if flag == 'train':
        x, y = x_data[:n_train], y_data[:n_train]
    elif flag == 'val':
        x, y = x_data[n_train : n_train + n_val], y_data[n_train : n_train + n_val]
    else:
        x, y = x_data[n_train + n_val :], y_data[n_train + n_val :]
        
    # Wrap into DataFrame for UnifiedDataset
    df = pd.DataFrame({
        'x': list(x),
        'observe_power': list(x), # Historical target for RevIN
        'observe_power_future': list(y),
    })
    
    data_set = UnifiedDataset(df)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return data_set, data_loader
