from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DLinear, PatchTST, iTransformer, TimesNet, FusionModel
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TimesNet': TimesNet,
        }
        
        if self.args.model == 'FusionModel':
            # Load base models
            base_models = {}
            model_names = ['DLinear', 'PatchTST', 'iTransformer', 'TimesNet']
            for name in model_names:
                m = model_dict[name](self.args.seq_len, self.args.pred_len).to(self.device)
                path = f'checkpoints/{name}.pth'
                if os.path.exists(path):
                    m.load_state_dict(torch.load(path, map_location=self.device))
                    print(f"Loaded weights for {name}")
                else:
                    print(f"Warning: {path} not found for FusionModel. Using randomly initialized weights.")
                base_models[name] = m
            
            model = FusionModel(base_models, self.args.seq_len, self.args.pred_len, self.args.enc_in, device=self.device).float()
        else:
            model = model_dict[self.args.model](self.args.seq_len, self.args.pred_len).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.model == 'FusionModel':
            # Optimize all trainable parameters (fusion layers)
            model_optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # Composite Loss for PV Power: Huber + Trend (1st-order difference)
        # Huber is robust to outliers (cloud cover spikes)
        # Trend loss handles distribution shifts by focusing on the change rate
        huber = nn.HuberLoss(delta=1.0)
        mse = nn.MSELoss()
        
        def composite_loss(pred, target):
            # 1. Base robust regression loss
            loss_val = huber(pred, target)
            
            # 2. Trend (Ramp) loss: focuses on the shape/change rate
            # Pred/Target shape: (B, P, C)
            if pred.shape[1] > 1:
                diff_pred = pred[:, 1:, :] - pred[:, :-1, :]
                diff_target = target[:, 1:, :] - target[:, :-1, :]
                loss_trend = mse(diff_pred, diff_target)
                return loss_val + 0.5 * loss_trend # lambda=0.5
            
            return loss_val
            
        return composite_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                outputs = self.model(batch)
                
                loss = criterion(outputs, batch['observe_power_future'])
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                if self.args.model == 'FusionModel':
                    outputs = self.model(batch)
                else:
                    outputs = self.model(batch)

                loss = criterion(outputs, batch['observe_power_future'])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                outputs = self.model(batch)

                pred = outputs.detach().cpu().numpy()
                true = batch['observe_power_future'].detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch['x'].detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
