import argparse
import os
import torch
import pandas as pd
import random
import numpy as np

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='FusionModel',
                        help='model name, options: [FusionModel, DLinear, PatchTST, iTransformer, TimesNet]')
    parser.add_argument('--fusion_version', type=str, default='base',
                        choices=['base', 'expert_head', 'multi_expert_head', 'legacy', 'v2', 'v3', 'v4', 'v5', 'tensor_v3'],
                        help='fusion model version selected by models/factory.py')
    parser.add_argument('--fusion_expert_name', type=str, default='m1',
                        choices=['m1', 'm2', 'm3', 'm4'],
                        help='single expert used by expert_head reconstruction')
    parser.add_argument('--fusion_d_model', type=int, default=None,
                        help='fusion hidden dimension override')
    parser.add_argument('--fusion_dropout', type=float, default=None,
                        help='fusion dropout override')
    parser.add_argument('--fusion_expert_dims', type=str, default=None,
                        help="expert hidden dims, e.g. 'm1:512,m2:256,m3:384,m4:512'")
    parser.add_argument('--fusion_loss', type=str, default=None,
                        choices=['mse', 'mae', 'huber'],
                        help='loss type for fusion versions that support it')
    parser.add_argument('--fusion_aux_loss_weight', type=float, default=None,
                        help='auxiliary expert-head loss weight for multi-head fusion')
    parser.add_argument('--target_key', type=str, default='observe_power_future',
                        help='target tensor key used by fusion models')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size (n_features)')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct start')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    from exp.exp_main import Exp_Main

    Exp = Exp_Main

    # Setting record of experiments
    setting = '{}_{}_{}_{}_sl{}_pl{}_lr{}_{}'.format(
        args.model_id,
        args.model,
        args.fusion_version,
        args.data,
        args.seq_len,
        args.pred_len,
        args.learning_rate,
        args.des)

    train_df = pd.read_parquet('xxx')
    valid_df = pd.read_parquet('xxx')
    test_df = pd.read_parquet('xxx')

    for col in ['observe_power', 'observe_power_future', 'chronos']:
        train_df[col] /= 500 
        valid_df[col] /= 500 
        test_df[col] /= 500 

    exp = Exp(args)  # set experiments

    if args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting, train_df, valid_df)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test_df)

        torch.cuda.empty_cache()
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test_df, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
