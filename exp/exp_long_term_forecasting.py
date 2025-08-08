from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        
        # 保存到txt文件（保持原有功能）
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        # 保存到CSV文件
        self._save_results_to_csv(setting, mae, mse, rmse, mape, mspe, dtw)

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def _save_results_to_csv(self, setting, mae, mse, rmse, mape, mspe, dtw):
        """
        将实验结果和所有参数保存到CSV文件中
        
        Args:
            setting (str): 实验设置标识符
            mae (float): 平均绝对误差
            mse (float): 均方误差
            rmse (float): 均方根误差
            mape (float): 平均绝对百分比误差
            mspe (float): 均方百分比误差
            dtw (float or str): DTW距离或'Not calculated'
        """
        # 创建结果字典，包含所有参数和实验结果
        result_dict = {
            # 实验基本信息
            'setting': setting,
            'task_name': self.args.task_name,
            'model_id': self.args.model_id,
            'model': self.args.model,
            'data': self.args.data,
            
            # 数据相关参数
            'root_path': self.args.root_path,
            'data_path': self.args.data_path,
            'features': self.args.features,
            'target': self.args.target,
            'freq': self.args.freq,
            
            # 预测相关参数
            'seq_len': self.args.seq_len,
            'label_len': self.args.label_len,
            'pred_len': self.args.pred_len,
            'seasonal_patterns': self.args.seasonal_patterns,
            'inverse': self.args.inverse,
            
            # 模型相关参数
            'expand': self.args.expand,
            'd_conv': self.args.d_conv,
            'top_k': self.args.top_k,
            'num_kernels': self.args.num_kernels,
            'enc_in': self.args.enc_in,
            'dec_in': self.args.dec_in,
            'c_out': self.args.c_out,
            'd_model': self.args.d_model,
            'n_heads': self.args.n_heads,
            'e_layers': self.args.e_layers,
            'd_layers': self.args.d_layers,
            'd_ff': self.args.d_ff,
            'moving_avg': self.args.moving_avg,
            'factor': self.args.factor,
            'distil': self.args.distil,
            'dropout': self.args.dropout,
            'embed': self.args.embed,
            'activation': self.args.activation,
            'channel_independence': self.args.channel_independence,
            'decomp_method': self.args.decomp_method,
            'use_norm': self.args.use_norm,
            'down_sampling_layers': self.args.down_sampling_layers,
            'down_sampling_window': self.args.down_sampling_window,
            'down_sampling_method': self.args.down_sampling_method,
            'seg_len': self.args.seg_len,
            
            # 训练相关参数
            'num_workers': self.args.num_workers,
            'itr': self.args.itr,
            'train_epochs': self.args.train_epochs,
            'batch_size': self.args.batch_size,
            'patience': self.args.patience,
            'learning_rate': self.args.learning_rate,
            'des': self.args.des,
            'loss': self.args.loss,
            'lradj': self.args.lradj,
            'use_amp': self.args.use_amp,
            
            # GPU相关参数
            'use_gpu': self.args.use_gpu,
            'gpu': self.args.gpu,
            'gpu_type': self.args.gpu_type,
            'use_multi_gpu': self.args.use_multi_gpu,
            'devices': self.args.devices,
            
            # 投影器参数
            'p_hidden_dims': str(self.args.p_hidden_dims),
            'p_hidden_layers': self.args.p_hidden_layers,
            
            # 指标相关参数
            'use_dtw': self.args.use_dtw,
            
            # 数据增强参数
            'augmentation_ratio': self.args.augmentation_ratio,
            'seed': self.args.seed,
            'jitter': self.args.jitter,
            'scaling': self.args.scaling,
            'permutation': self.args.permutation,
            'randompermutation': self.args.randompermutation,
            'magwarp': self.args.magwarp,
            'timewarp': self.args.timewarp,
            'windowslice': self.args.windowslice,
            'windowwarp': self.args.windowwarp,
            'rotation': self.args.rotation,
            'spawner': self.args.spawner,
            'dtwwarp': self.args.dtwwarp,
            'shapedtwwarp': self.args.shapedtwwarp,
            'wdba': self.args.wdba,
            'discdtw': self.args.discdtw,
            'discsdtw': self.args.discsdtw,
            'extra_tag': self.args.extra_tag,
            
            # TimeXer参数
            'patch_len': self.args.patch_len,
            
            # 实验结果
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe,
            'dtw': dtw
        }
        
        # CSV文件路径
        csv_file = './log/long_term_forecast/experiment_results.csv'
        
        # 确保目录存在
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        # 创建新的DataFrame
        df_new = pd.DataFrame([result_dict])
        
        # 检查文件是否存在且格式正确
        if os.path.exists(csv_file):
            try:
                # 尝试读取现有数据
                df_existing = pd.read_csv(csv_file)
                # 检查是否有重复列
                if len(df_existing.columns) != len(set(df_existing.columns)):
                    print(f"警告：CSV文件 {csv_file} 包含重复列，将重新创建文件")
                    df_existing = pd.DataFrame()
            except Exception as e:
                print(f"读取现有CSV文件时出错：{e}，将重新创建文件")
                df_existing = pd.DataFrame()
        else:
            df_existing = pd.DataFrame()
        
        # 以setting为主键进行更新或添加
        if not df_existing.empty:
            # 检查是否存在相同的setting
            existing_setting = df_existing['setting'].values
            new_setting = result_dict['setting']
            
            if new_setting in existing_setting:
                # 如果存在相同的setting，则更新该行
                setting_index = df_existing[df_existing['setting'] == new_setting].index[0]
                for col in df_new.columns:
                    df_existing.loc[setting_index, col] = df_new.iloc[0][col]
                df_combined = df_existing
                print(f'更新实验设置: {new_setting}')
            else:
                # 如果不存在相同的setting，则添加新行
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                print(f'添加新实验设置: {new_setting}')
        else:
            # 如果文件不存在或为空，直接使用新数据
            df_combined = df_new
            print(f'创建新实验设置: {result_dict["setting"]}')
        
        # 保存到CSV文件
        df_combined.to_csv(csv_file, index=False)
        print(f'实验结果已保存到 {csv_file}')
