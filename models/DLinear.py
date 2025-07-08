"""
Time-Series Decomposition Linear Model (DLinear)

输入张量统一形状：
    (B, L, D)
    B: batch_size
    L: seq_len，输入序列长度
    D: enc_in，变量数量

支持任务：
    • long_term_forecast / short_term_forecast
    • imputation
    • anomaly_detection
    • classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    DLinear 主模型类。

    该类封装了 DLinear 的全部逻辑，包括序列分解、趋势与季节性分支，以及针对不同任务的前向路径。
    算法来源：`Revisiting Non-Stationary Time Series Forecasting: A Seasonal-Trend Decomposition Based Architecture`
    (ICLR 2023)。

    Attributes:
        task_name (str): 任务名称，支持 `long_term_forecast`、`short_term_forecast`、
            `imputation`、`anomaly_detection`、`classification`。
        seq_len (int): 输入序列长度 \(L)。
        pred_len (int): 预测序列长度 \(O)。对于分类 / 异常检测 / 插值任务等同于 `seq_len`。
        decompsition (nn.Module): Autoformer 的时间序列分解模块，用于拆分趋势与季节性。
        individual (bool): 若为 `True`，则为每个变量创建独立线性层；否则共享线性层。
        channels (int): 变量数量 \(D)。
        Linear_Seasonal (nn.Linear | nn.ModuleList): 季节性分支线性层。
        Linear_Trend (nn.Linear | nn.ModuleList): 趋势分支线性层。
        projection (nn.Linear, optional): 分类任务的最终映射层。

    Example:
        ```python
        from easydict import EasyDict as edict

        cfg = edict({
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'moving_avg': 25,
            'enc_in': 7,
            'num_class': 3,
        })

        model = Model(cfg, individual=True)
        x = torch.randn(32, cfg.seq_len, cfg.enc_in)
        pred = model(x, None, None, None)  # pred.shape == (32, cfg.pred_len, cfg.enc_in)
        ```
    """

    def __init__(self, configs, individual=False):
        """初始化 DLinear 模型。

        Args:
            configs: 配置对象，需包含 `task_name`, `seq_len`, `pred_len`, `moving_avg`,
                `enc_in`，以及 `num_class`（若为分类任务）。
            individual (bool, optional): `True` 表示为每个变量建立独立线性层，
                `False` 表示所有变量共享线性层。
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        """编码器：分解输入序列并执行线性建模。

        Args:
            x (torch.Tensor): 输入序列，形状 `(B, L, D)`。

        Returns:
            torch.Tensor: 形状 `(B, O, D)` 的编码输出，其中 `O = pred_len`。
        """
        # (1) 序列分解 ➡️ 得到 seasonal 与 trend，两者形状均为 (B, L, D)
        seasonal_init, trend_init = self.decompsition(x)

        # (2) 调整维度以按变量进行处理： (B, L, D) -> (B, D, L)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            # (3) individual=True: 为每个变量使用独立线性层
            seasonal_output = torch.zeros(
                (seasonal_init.size(0), seasonal_init.size(1), self.pred_len),
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                (trend_init.size(0), trend_init.size(1), self.pred_len),
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for i in range(self.channels):
                # (4) 逐变量映射: (B, L) -> (B, O)
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # (3') individual=False: 使用共享线性层 (B, D, L) -> (B, D, O)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        # (5) 合并趋势与季节性分量 (B, D, O)
        x = seasonal_output + trend_output

        # (6) 重新调整为 (B, O, D) 供后续模块使用
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        """长 / 短期预测接口。

        Args:
            x_enc (torch.Tensor): 输入序列 `(B, L, D)`。

        Returns:
            torch.Tensor: 预测结果 `(B, O, D)`。
        """
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        """插值任务接口，同 `encoder`。"""
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        """异常检测任务接口，同 `encoder`。"""
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        """分类任务接口。

        Args:
            x_enc (torch.Tensor): 输入序列 `(B, L, D)`。

        Returns:
            torch.Tensor: 分类 logits，形状 `(B, num_classes)`。
        """
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """统一前向接口，根据 `task_name` 调度不同分支。"""
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
