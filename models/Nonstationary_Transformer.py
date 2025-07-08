"""
非平稳时间序列 Transformer 模型实现

该模块基于论文 “Non-stationary Transformer: Exploring De-stationary Transformer for Long-Term Time-Series Forecasting”
实现了一个可同时用于预测、插值、异常检测与分类的统一模型。

主要组件：
1. Projector: 学习样本级去平稳化因子 (tau, delta)。
2. Encoder / Decoder: 基于去平稳化注意力 (DSAttention) 的 Transformer 编码器/解码器。
3. Model: 根据任务类型 (long_term_forecast、imputation 等) 调用相应前向逻辑。

论文链接: https://openreview.net/pdf?id=ucNDIDRNjjv
"""
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F


class Projector(nn.Module):
    '''
    Projector 模块

    该模块首先通过一维卷积 `series_conv` 在时间维度上聚合序列信息，然后将卷积输出与样本统计量
    (均值或标准差) 拼接，并交由多层感知机 (MLP) 预测去平稳化因子。

    输入:
        x     -- 张量形状 (B, S, E)，原始时间序列片段
        stats -- 张量形状 (B, 1, E)，对应样本的统计量 (均值或标准差)

    输出:
        y     -- 张量形状 (B, O)，其中 O 由 `output_dim` 指定，可为标量 (tau) 或向量 (delta)

    论文: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        """
        前向传播

        参数:
            x (Tensor): 原始时间序列片段，形状 [B, S, E]
            stats (Tensor): 样本统计量(均值或标准差)，形状 [B, 1, E]

        返回:
            Tensor: 预测的去平稳化因子，形状 [B, O]
        """
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            DSAttention(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        长期/短期预测任务的前向传播逻辑。

        具体流程:
        1. 对输入序列进行样本级归一化，得到零均值单位方差序列；
        2. 通过 Projector 分别预测方差缩放因子 tau(经 exp 保证为正) 及均值偏移向量 delta；
        3. 构造解码器输入: label_len 段已知历史 + pred_len 段全零占位；
        4. 在去平稳化空间运行编码器与解码器；
        5. 将解码器输出按均值/方差反标准化回原始尺度。

        参数:
            x_enc (Tensor): 编码器输入序列，形状 [B, seq_len, E]
            x_mark_enc (Tensor): 编码器时间特征，形状 [B, seq_len, F] 或 None
            x_dec (Tensor): 解码器输入(已知历史 + 占位)，形状 [B, label_len+pred_len, E]
            x_mark_dec (Tensor): 解码器时间特征，形状 [B, label_len+pred_len, F]

        返回:
            Tensor: 解码器完整输出(含已知+预测)，形状 [B, label_len+pred_len, E]
        """
        x_raw = x_enc.clone().detach()

        # 1) 对输入序列进行标准化: 减去均值再除以标准差，得到零均值单位方差序列
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        
        # 2) 通过 Projector 估计方差缩放因子 tau (正标量)，并做指数变换保证其为正
        tau = self.tau_learner(x_raw, std_enc)
        threshold = 80.0
        tau_clamped = torch.clamp(tau, max=threshold)  # avoid numerical overflow
        tau = tau_clamped.exp()
        
        # 3) 通过 Projector 估计均值偏移向量 delta
        delta = self.delta_learner(x_raw, mean_enc)

        # 4) 构造解码器输入: 先拼接 label_len 段已知历史，再补零占位预测步长
        x_dec_new = torch.cat([
            x_enc[:, -self.label_len:, :],  # 已知部分
            torch.zeros_like(x_dec[:, -self.pred_len:, :])  # 预测占位
        ], dim=1).to(x_enc.device).clone()

        # 5) 编码器前向传播
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        # 6) 解码器前向传播并将输出反标准化回原始尺度
        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta)
        dec_out = dec_out * std_enc + mean_enc  # 反标准化
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        缺失值填充任务前向传播逻辑。

        与 forecast 相似，但输入序列包含缺失值并由 mask 指示有效位置。
        步骤:
        1. 基于有效元素计算样本级均值/方差并归一化；
        2. 预测 tau / delta 去平稳化因子；
        3. 使用 Encoder 输出通过投影层得到填充值；
        4. 输出反归一化回原始尺度。

        参数:
            x_enc (Tensor): 输入序列，形状 [B, seq_len, E]
            x_mark_enc (Tensor): 编码器时间特征，形状 [B, seq_len, F]
            x_dec (Tensor): 占位参数，未使用
            x_mark_dec (Tensor): 占位参数，未使用
            mask (Tensor): 缺失值掩码，1 表示有效，0 表示缺失，形状 [B, seq_len, E]

        返回:
            Tensor: 填充后的完整序列，形状 [B, seq_len, E]
        """
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        mean_enc = mean_enc.unsqueeze(1).detach()
        x_enc = x_enc - mean_enc
        x_enc = x_enc.masked_fill(mask == 0, 0)
        std_enc = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        std_enc = std_enc.unsqueeze(1).detach()
        x_enc /= std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc)
        threshold = 80.0
        tau_clamped = torch.clamp(tau, max=threshold)  # avoid numerical overflow
        tau = tau_clamped.exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.projection(enc_out)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        异常检测任务前向传播逻辑。

        模型试图重建输入序列，重建误差可用于后续的异常判别。

        参数:
            x_enc (Tensor): 输入序列，形状 [B, seq_len, E]

        返回:
            Tensor: 重建序列，形状 [B, seq_len, E]
        """
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc)
        threshold = 80.0
        tau_clamped = torch.clamp(tau, max=threshold)  # avoid numerical overflow
        tau = tau_clamped.exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.projection(enc_out)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        分类任务前向传播逻辑。

        1. Encoder 提取序列特征；
        2. 应用 GELU + Dropout；
        3. 展平后映射到类别空间。

        参数:
            x_enc (Tensor): 输入序列，形状 [B, seq_len, E]
            x_mark_enc (Tensor): 有效位置信息(同 padding mask)，形状 [B, seq_len]

        返回:
            Tensor: 分类 logits，形状 [B, num_class]
        """
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        std_enc = torch.sqrt(
            torch.var(x_enc - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc)
        threshold = 80.0
        tau_clamped = torch.clamp(tau, max=threshold)  # avoid numerical overflow
        tau = tau_clamped.exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        根据 task_name 调度到对应的任务分支。

        支持的任务:
            long_term_forecast / short_term_forecast: 调用 forecast 并返回最后 pred_len 步预测结果；
            imputation: 调用 imputation 填补缺失值；
            anomaly_detection: 调用 anomaly_detection 重建序列；
            classification: 调用 classification 进行序列级分类。
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        return None
