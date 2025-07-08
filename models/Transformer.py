# 导入必要的PyTorch模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入自定义的Transformer编码器和解码器层
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
# 导入自注意力机制相关模块
from layers.SelfAttention_Family import FullAttention, AttentionLayer
# 导入数据嵌入模块
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    原始Transformer模型
    时间复杂度为O(L^2)
    基于论文: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    
    该模型支持多种时间序列任务:
    - 长期预测 (long_term_forecast)
    - 短期预测 (short_term_forecast)
    - 缺失值填充 (imputation)
    - 异常检测 (anomaly_detection)
    - 分类 (classification)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # 保存任务类型和预测长度
        self.task_name = configs.task_name  # 任务类型
        self.pred_len = configs.pred_len    # 预测长度
        
        # 编码器嵌入层 - 将输入数据转换为模型可处理的高维特征
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        # 构建Transformer编码器
        # 编码器负责学习输入序列的表示
        self.encoder = Encoder(
            [
                # 创建多个编码器层
                EncoderLayer(
                    # 自注意力层 - 允许序列中的每个位置关注其他所有位置
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,     # 模型维度
                    configs.d_ff,        # 前馈网络维度
                    dropout=configs.dropout,      # Dropout率
                    activation=configs.activation # 激活函数
                ) for l in range(configs.e_layers)  # 编码器层数
            ],
            norm_layer=nn.LayerNorm(configs.d_model)  # 层归一化
        )
        
        # 根据任务类型构建解码器
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 预测任务需要解码器
            # 解码器嵌入层 - 处理目标序列
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            # 构建Transformer解码器
            self.decoder = Decoder(
                [
                    # 创建多个解码器层
                    DecoderLayer(
                        # 自注意力层 - 目标序列内部的注意力(带掩码)
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        # 交叉注意力层 - 目标序列与编码器输出之间的注意力
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,     # 模型维度
                        configs.d_ff,        # 前馈网络维度
                        dropout=configs.dropout,      # Dropout率
                        activation=configs.activation, # 激活函数
                    )
                    for l in range(configs.d_layers)  # 解码器层数
                ],
                norm_layer=nn.LayerNorm(configs.d_model),  # 层归一化
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)  # 输出投影层
            )
        
        # 缺失值填充任务的输出投影层
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
        # 异常检测任务的输出投影层
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
        # 分类任务的特殊处理
        if self.task_name == 'classification':
            self.act = F.gelu  # GELU激活函数
            self.dropout = nn.Dropout(configs.dropout)  # Dropout层
            # 分类投影层：将序列特征展平后映射到类别数量
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        预测任务的前向传播函数
        
        参数:
            x_enc: 编码器输入序列 [batch_size, seq_len, features]
            x_mark_enc: 编码器时间标记 [batch_size, seq_len, mark_features]
            x_dec: 解码器输入序列 [batch_size, label_len+pred_len, features]
            x_mark_dec: 解码器时间标记 [batch_size, label_len+pred_len, mark_features]
        
        返回:
            dec_out: 解码器输出 [batch_size, label_len+pred_len, features]
        """
        # 编码器嵌入：将输入数据和时间标记转换为高维特征
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # 编码器前向传播：学习输入序列的表示
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 解码器嵌入：处理目标序列
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # 解码器前向传播：基于编码器输出生成预测
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        缺失值填充任务的前向传播函数
        
        参数:
            x_enc: 编码器输入序列（包含缺失值） [batch_size, seq_len, features]
            x_mark_enc: 编码器时间标记 [batch_size, seq_len, mark_features]
            x_dec: 解码器输入序列（未使用）
            x_mark_dec: 解码器时间标记（未使用）
            mask: 缺失值掩码 [batch_size, seq_len, features]
        
        返回:
            dec_out: 填充后的序列 [batch_size, seq_len, features]
        """
        # 编码器嵌入：将输入数据和时间标记转换为高维特征
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # 编码器前向传播：学习序列表示，用于缺失值填充
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 通过投影层输出填充结果
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        异常检测任务的前向传播函数
        
        参数:
            x_enc: 编码器输入序列 [batch_size, seq_len, features]
        
        返回:
            dec_out: 异常检测结果 [batch_size, seq_len, features]
        """
        # 编码器嵌入：将输入数据转换为高维特征（无时间标记）
        enc_out = self.enc_embedding(x_enc, None)
        # 编码器前向传播：学习序列表示，用于异常检测
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 通过投影层输出异常检测结果
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        分类任务的前向传播函数
        
        参数:
            x_enc: 编码器输入序列 [batch_size, seq_len, features]
            x_mark_enc: 编码器时间标记 [batch_size, seq_len, mark_features]
        
        返回:
            output: 分类结果 [batch_size, num_classes]
        """
        # 编码器嵌入：将输入数据转换为高维特征（无时间标记）
        enc_out = self.enc_embedding(x_enc, None)
        # 编码器前向传播：学习序列表示
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 输出处理
        # 应用GELU激活函数（Transformer编码器输出没有非线性）
        output = self.act(enc_out)  
        # 应用Dropout防止过拟合
        output = self.dropout(output)
        # 使用时间标记进行零填充嵌入（mask padding positions）
        output = output * x_mark_enc.unsqueeze(-1)  
        # 重塑张量：将序列特征展平 (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)  
        # 通过投影层输出分类结果 (batch_size, num_classes)
        output = self.projection(output)  
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        模型的主要前向传播函数，根据任务类型调用相应的子函数
        
        参数:
            x_enc: 编码器输入序列 [batch_size, seq_len, features]
            x_mark_enc: 编码器时间标记 [batch_size, seq_len, mark_features]
            x_dec: 解码器输入序列 [batch_size, label_len+pred_len, features]
            x_mark_dec: 解码器时间标记 [batch_size, label_len+pred_len, mark_features]
            mask: 缺失值掩码（仅用于imputation任务）
        
        返回:
            根据任务类型返回相应的输出:
            - 预测任务: [batch_size, pred_len, features]
            - 缺失值填充: [batch_size, seq_len, features]
            - 异常检测: [batch_size, seq_len, features]
            - 分类: [batch_size, num_classes]
        """
        # 长期或短期预测任务
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 只返回预测部分（最后pred_len个时间步）
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
        # 缺失值填充任务
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        
        # 异常检测任务
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        
        # 分类任务
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        
        # 未知任务类型
        return None
