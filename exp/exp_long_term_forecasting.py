# 导入必要的库
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

# 忽略警告信息
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    长时序预测实验类，继承自基础实验类 Exp_Basic。
    该类封装了长时序预测任务的整个流程，包括模型构建、数据加载、训练、验证和测试。
    """
    def __init__(self, args):
        """
        初始化实验。

        Args:
            args (Namespace): 包含所有实验参数的命名空间对象。
        """
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        """
        构建模型。
        根据参数选择指定的模型，并进行初始化。
        如果启用了多GPU，则使用 DataParallel 进行包装。

        Returns:
            torch.nn.Module: 构建好的模型实例。
        """
        # 从模型字典中获取模型类并实例化
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 如果使用多GPU，则使用DataParallel进行包装
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        获取数据集和数据加载器。

        Args:
            flag (str): 'train', 'val' 或 'test'，用于指定获取哪部分数据。

        Returns:
            tuple: (数据集, 数据加载器)
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        选择优化器。

        Returns:
            torch.optim.Optimizer: 优化器实例。
        """
        # 使用Adam优化器
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        选择损失函数。

        Returns:
            torch.nn.Module: 损失函数实例。
        """
        # 使用均方误差（MSE）作为损失函数
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        验证过程。
        在验证集上评估模型性能。

        Args:
            vali_data (Dataset): 验证数据集。
            vali_loader (DataLoader): 验证数据加载器。
            criterion (torch.nn.Module): 损失函数。

        Returns:
            float: 在验证集上的平均损失。
        """
        total_loss = []
        # 将模型设置为评估模式
        self.model.eval()
        with torch.no_grad():  # 禁用梯度计算
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # 将数据移动到指定设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 准备解码器输入
                # 解码器输入的前半部分是标签序列，后半部分用0填充
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 模型前向传播
                if self.args.use_amp:  # 使用自动混合精度
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # 根据特征类型（多变量或单变量）确定输出维度
                f_dim = -1 if self.args.features == 'MS' else 0
                # 取输出的预测部分
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # 取真实值的预测部分
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # 计算损失
                loss = criterion(pred, true)
                total_loss.append(loss.item())

        # 计算平均损失
        total_loss = np.average(total_loss)
        # 将模型设置回训练模式
        self.model.train()
        return total_loss

    def train(self, setting):
        """
        训练过程。
        执行完整的模型训练流程，包括训练、验证、早停和学习率调整。

        Args:
            setting (str): 用于标识本次实验设置的字符串，通常用于命名保存路径。

        Returns:
            torch.nn.Module: 训练好的模型。
        """
        # 获取训练、验证和测试数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 设置模型检查点保存路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        # 初始化早停机制
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 选择优化器和损失函数
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 如果使用自动混合精度，则初始化GradScaler
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 开始训练循环
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # 将数据移动到指定设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 准备解码器输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 模型前向传播和损失计算
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

                # 每100次迭代打印一次训练信息
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                # 反向传播和参数更新
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            
            # 计算并打印训练、验证和测试损失
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            # 执行早停检查
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        测试过程。
        在测试集上评估最终模型的性能，并保存结果。

        Args:
            setting (str): 用于标识本次实验设置的字符串。
            test (int): 如果为1，则从文件中加载模型进行测试。
        """
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

                # 准备解码器输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 模型前向传播
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
                
                # 如果数据经过了标准化，则进行逆标准化
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    # 如果输出和真实值的特征维度不匹配，则平铺输出以匹配
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
                
                # 每20次迭代保存一次可视化结果
                if i % 20 == 0:
                    input_data = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_data.shape
                        input_data = test_data.inverse_transform(input_data.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    # 将输入和真实值/预测值拼接，用于可视化
                    gt = np.concatenate((input_data[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_data[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 拼接所有批次的预测和真实值
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 计算DTW距离
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
            dtw_metric_val = np.array(dtw_list).mean()
        else:
            dtw_metric_val = 'Not calculated'

        # 计算标准评估指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, dtw:{dtw_metric_val}')
        
        # 将结果写入文件
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write(f'mse:{mse}, mae:{mae}, dtw:{dtw_metric_val}')
            f.write('\n')
            f.write('\n')

        # 保存指标、预测值和真实值
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
