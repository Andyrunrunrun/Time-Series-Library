import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    该模型实现了iTransformer，基于论文: https://arxiv.org/abs/2310.06625
    
    该模型支持多种时间序列任务:
    - 长期预测 (long_term_forecast)
    - 短期预测 (short_term_forecast)
    - 缺失值填充 (imputation)
    - 异常检测 (anomaly_detection)
    - 分类 (classification)
    
    Args:
        configs: 配置对象，包含模型的超参数和任务类型。
    """

    def __init__(self, configs):
        """
        初始化iTransformer模型。

        Args:
            configs: 配置对象，包含以下属性：
                - task_name (str): 任务类型。
                - seq_len (int): 输入序列长度。
                - pred_len (int): 预测序列长度。
                - d_model (int): 模型维度。
                - embed (str): 嵌入类型。
                - freq (str): 时间频率。
                - dropout (float): Dropout率。
                - factor (int): 注意力因子。
                - n_heads (int): 注意力头数。
                - d_ff (int): 前馈网络维度。
                - e_layers (int): 编码器层数。
                - activation (str): 激活函数。
                - enc_in (int): 编码器输入维度。
                - num_class (int): 分类任务的类别数。
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # 嵌入层
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # 编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # 解码器
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        预测任务的前向传播函数。

        Args:
            x_enc (torch.Tensor): 编码器输入序列，形状为 (batch_size, seq_len, features)。
            x_mark_enc (torch.Tensor): 编码器时间标记，形状为 (batch_size, seq_len, mark_features)。
            x_dec (torch.Tensor): 解码器输入序列，形状为 (batch_size, label_len+pred_len, features)。
            x_mark_dec (torch.Tensor): 解码器时间标记，形状为 (batch_size, label_len+pred_len, mark_features)。

        Returns:
            torch.Tensor: 解码器输出，形状为 (batch_size, pred_len, features)。
        """
        # 非平稳Transformer的归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # 嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # 非平稳Transformer的反归一化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        缺失值填充任务的前向传播函数。

        Args:
            x_enc (torch.Tensor): 编码器输入序列，形状为 (batch_size, seq_len, features)。
            x_mark_enc (torch.Tensor): 编码器时间标记，形状为 (batch_size, seq_len, mark_features)。
            x_dec (torch.Tensor): 解码器输入序列，形状为 (batch_size, label_len+pred_len, features)。
            x_mark_dec (torch.Tensor): 解码器时间标记，形状为 (batch_size, label_len+pred_len, mark_features)。
            mask (torch.Tensor): 缺失值掩码，形状为 (batch_size, seq_len, features)。

        Returns:
            torch.Tensor: 填充后的序列，形状为 (batch_size, seq_len, features)。
        """
        # 非平稳Transformer的归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # 嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # 非平稳Transformer的反归一化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        异常检测任务的前向传播函数。

        Args:
            x_enc (torch.Tensor): 编码器输入序列，形状为 (batch_size, seq_len, features)。

        Returns:
            torch.Tensor: 异常检测结果，形状为 (batch_size, seq_len, features)。
        """
        # 非平稳Transformer的归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # 嵌入
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # 非平稳Transformer的反归一化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        分类任务的前向传播函数。

        Args:
            x_enc (torch.Tensor): 编码器输入序列，形状为 (batch_size, seq_len, features)。
            x_mark_enc (torch.Tensor): 编码器时间标记，形状为 (batch_size, seq_len, mark_features)。

        Returns:
            torch.Tensor: 分类结果，形状为 (batch_size, num_classes)。
        """
        # 嵌入
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 输出
        output = self.act(enc_out)  # Transformer编码器/解码器输出不包含非线性
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        模型的主要前向传播函数，根据任务类型调用相应的子函数。

        Args:
            x_enc (torch.Tensor): 编码器输入序列，形状为 (batch_size, seq_len, features)。
            x_mark_enc (torch.Tensor): 编码器时间标记，形状为 (batch_size, seq_len, mark_features)。
            x_dec (torch.Tensor): 解码器输入序列，形状为 (batch_size, label_len+pred_len, features)。
            x_mark_dec (torch.Tensor): 解码器时间标记，形状为 (batch_size, label_len+pred_len, mark_features)。
            mask (torch.Tensor, optional): 缺失值掩码，仅用于imputation任务。

        Returns:
            torch.Tensor: 根据任务类型返回相应的输出。
            - 预测任务: 形状为 (batch_size, pred_len, features)
            - 缺失值填充: 形状为 (batch_size, seq_len, features)
            - 异常检测: 形状为 (batch_size, seq_len, features)
            - 分类: 形状为 (batch_size, num_classes)
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
            return dec_out  # [B, N]
        return None
