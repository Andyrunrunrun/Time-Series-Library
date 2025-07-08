import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class FlattenHead(nn.Module):
    """
    预测头部模块 - 将编码器输出映射为最终预测结果
    
    该模块负责将多维特征表示展平并映射到目标预测长度。它是TimeXer模型的最后一层，
    用于生成具体的预测值。设计思路来源于Vision Transformer中的分类头部，
    但针对时间序列预测任务进行了适配。
    
    主要功能:
    - 将四维特征张量展平为二维
    - 通过线性变换映射到目标预测长度
    - 应用dropout防止过拟合
    
    Args:
        n_vars (int): 变量数量（时间序列的特征维度）
        nf (int): 输入特征数量，通常为 d_model * (patch_num + 1)
        target_window (int): 目标预测长度
        head_dropout (float, optional): dropout概率，默认为0
    """
    
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        # (1) 展平操作：将最后两个维度合并 [bs, nvars, d_model, patch_num] -> [bs, nvars, d_model*patch_num]
        self.flatten = nn.Flatten(start_dim=-2)
        # (2) 线性映射：将特征维度映射到预测长度 [bs, nvars, nf] -> [bs, nvars, target_window]
        self.linear = nn.Linear(nf, target_window)
        # (3) dropout正则化
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        预测头部的前向传播
        
        Args:
            x (torch.Tensor): 编码器输出特征
                形状: [batch_size, n_vars, d_model, patch_num]
        
        Returns:
            torch.Tensor: 预测结果
                形状: [batch_size, n_vars, target_window]
                
        张量形状变换流程:
            [bs, nvars, d_model, patch_num] 
            -> flatten -> [bs, nvars, d_model*patch_num]
            -> linear -> [bs, nvars, target_window]
            -> dropout -> [bs, nvars, target_window]
        """
        # (1) 展平最后两个维度：[bs, nvars, d_model, patch_num] -> [bs, nvars, d_model*patch_num]
        x = self.flatten(x)
        # (2) 线性变换映射到预测长度：[bs, nvars, d_model*patch_num] -> [bs, nvars, target_window]
        x = self.linear(x)
        # (3) 应用dropout防止过拟合
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    """
    编码器嵌入层 - 时间序列分块嵌入与位置编码
    
    该模块实现了TimeXer的核心创新之一：时间序列分块(patching)机制。
    受Vision Transformer启发，将连续的时间序列切分成多个patch，
    每个patch作为一个token进行处理。同时添加全局token和位置编码。
    
    理论依据:
    - Vision Transformer的patch embedding思想
    - 分块处理可以减少序列长度，提高计算效率
    - 全局token用于捕获全局信息
    
    Args:
        n_vars (int): 变量数量（时间序列的特征维度）
        d_model (int): 模型嵌入维度
        patch_len (int): 每个patch的长度
        dropout (float): dropout概率
    """
    
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # 保存patch长度用于分块操作
        self.patch_len = patch_len

        # (1) 值嵌入：将每个patch映射到d_model维度 [patch_len] -> [d_model]
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # (2) 全局token：可学习参数，用于捕获全局信息 [1, n_vars, 1, d_model]
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        # (3) 位置编码：为每个patch位置提供位置信息
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        编码器嵌入的前向传播
        
        Args:
            x (torch.Tensor): 输入时间序列
                形状: [batch_size, n_vars, seq_len]
        
        Returns:
            tuple: (embedded_sequences, n_vars)
                embedded_sequences (torch.Tensor): 嵌入后的序列
                    形状: [batch_size * n_vars, patch_num + 1, d_model]
                n_vars (int): 变量数量
                
        张量形状变换流程:
            输入: [B, n_vars, L]
            -> unfold -> [B, n_vars, patch_num, patch_len]
            -> reshape -> [B*n_vars, patch_num, patch_len]
            -> value_embedding -> [B*n_vars, patch_num, d_model]
            -> position_embedding -> [B*n_vars, patch_num, d_model]
            -> reshape -> [B, n_vars, patch_num, d_model]
            -> concat global token -> [B, n_vars, patch_num+1, d_model]
            -> reshape -> [B*n_vars, patch_num+1, d_model]
        """
        # (1) 获取变量数量和准备全局token
        n_vars = x.shape[1]
        # 复制全局token到当前batch大小：[1, n_vars, 1, d_model] -> [B, n_vars, 1, d_model]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        # (2) 时间序列分块操作 (patching)
        # unfold操作：将序列按patch_len切分 [B, n_vars, L] -> [B, n_vars, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # 重塑为二维以便批量处理：[B, n_vars, patch_num, patch_len] -> [B*n_vars, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        # (3) 值嵌入和位置编码
        # 值嵌入：将每个patch映射到d_model维度 [B*n_vars, patch_num, patch_len] -> [B*n_vars, patch_num, d_model]
        # 位置编码：为每个patch添加位置信息 [B*n_vars, patch_num, d_model] -> [B*n_vars, patch_num, d_model]
        x = self.value_embedding(x) + self.position_embedding(x)
        
        # (4) 重塑回四维以便添加全局token
        # [B*n_vars, patch_num, d_model] -> [B, n_vars, patch_num, d_model]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        # 在patch维度上连接全局token：[B, n_vars, patch_num, d_model] + [B, n_vars, 1, d_model] -> [B, n_vars, patch_num+1, d_model]
        x = torch.cat([x, glb], dim=2)
        
        # (5) 最终重塑为三维以输入到Transformer
        # [B, n_vars, patch_num+1, d_model] -> [B*n_vars, patch_num+1, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    """
    Transformer编码器 - 多层编码器的堆叠
    
    标准的Transformer编码器结构，由多个EncoderLayer层堆叠而成。
    支持可选的层归一化和输出投影层。
    
    Args:
        layers (list): EncoderLayer层的列表
        norm_layer (nn.Module, optional): 层归一化模块，默认为None
        projection (nn.Module, optional): 输出投影层，默认为None
    """
    
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        # 编码器层列表
        self.layers = nn.ModuleList(layers)
        # 可选的层归一化
        self.norm = norm_layer
        # 可选的输出投影层
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        编码器前向传播
        
        Args:
            x (torch.Tensor): 主输入序列，形状: [B*n_vars, patch_num+1, d_model]
            cross (torch.Tensor): 交叉注意力的键值序列，形状: [B, seq_len, d_model]
            x_mask (torch.Tensor, optional): 自注意力掩码
            cross_mask (torch.Tensor, optional): 交叉注意力掩码
            tau (float, optional): 温度参数
            delta (float, optional): 增量参数
        
        Returns:
            torch.Tensor: 编码器输出，形状: [B*n_vars, patch_num+1, d_model]
        """
        # (1) 逐层通过编码器层
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        # (2) 可选的层归一化
        if self.norm is not None:
            x = self.norm(x)

        # (3) 可选的输出投影
        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    """
    TimeXer编码器层 - 自注意力 + 交叉注意力 + 前馈网络
    
    TimeXer的创新编码器层设计，包含：
    1. 自注意力机制：处理patch序列内部的依赖关系
    2. 交叉注意力机制：全局token与外部序列的交互
    3. 前馈网络：非线性变换和特征提取
    
    特点:
    - 全局token设计：最后一个token作为全局信息聚合点
    - 交叉注意力：全局token与外部序列进行交互
    - 残差连接和层归一化：稳定训练过程
    
    Args:
        self_attention (AttentionLayer): 自注意力层
        cross_attention (AttentionLayer): 交叉注意力层
        d_model (int): 模型维度
        d_ff (int, optional): 前馈网络隐藏层维度，默认为4*d_model
        dropout (float): dropout概率，默认为0.1
        activation (str): 激活函数类型，默认为"relu"
    """
    
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # 前馈网络：两个1D卷积层实现 [d_model] -> [d_ff] -> [d_model]
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # 三个层归一化：分别用于自注意力、交叉注意力、前馈网络
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        编码器层前向传播
        
        Args:
            x (torch.Tensor): 输入序列，形状: [B*n_vars, patch_num+1, d_model]
            cross (torch.Tensor): 交叉注意力的键值序列，形状: [B, seq_len, d_model]
            x_mask (torch.Tensor, optional): 自注意力掩码
            cross_mask (torch.Tensor, optional): 交叉注意力掩码
            tau (float, optional): 温度参数
            delta (float, optional): 增量参数
        
        Returns:
            torch.Tensor: 编码器层输出，形状: [B*n_vars, patch_num+1, d_model]
            
        张量形状变换流程:
            (1) 自注意力: [B*n_vars, patch_num+1, d_model] -> [B*n_vars, patch_num+1, d_model]
            (2) 提取全局token: [B*n_vars, patch_num+1, d_model] -> [B*n_vars, 1, d_model]
            (3) 交叉注意力: [B, n_vars, d_model] @ [B, seq_len, d_model] -> [B, n_vars, d_model]
            (4) 前馈网络: [B*n_vars, patch_num+1, d_model] -> [B*n_vars, patch_num+1, d_model]
        """
        B, L, D = cross.shape
        
        # (1) 自注意力模块 + 残差连接 + 层归一化
        # 形状保持: [B*n_vars, patch_num+1, d_model] -> [B*n_vars, patch_num+1, d_model]
        x = x + self.dropout(self.self_attention(
            x, x, x,  # Q, K, V都来自同一个输入
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # (2) 交叉注意力模块：全局token与外部序列交互
        # 提取全局token（最后一个token）: [B*n_vars, patch_num+1, d_model] -> [B*n_vars, 1, d_model]
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        # 重塑为批量处理格式: [B*n_vars, 1, d_model] -> [B, n_vars, d_model]
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        
        # 交叉注意力：全局token作为Q，外部序列作为K和V
        # [B, n_vars, d_model] 与 [B, seq_len, d_model] 进行交叉注意力
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,  # Q来自全局token，K和V来自外部序列
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        
        # 重塑回原始格式: [B, n_vars, d_model] -> [B*n_vars, 1, d_model]
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        # 残差连接和层归一化
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        # (3) 重新组合：将更新后的全局token拼接回原序列
        # [B*n_vars, patch_num, d_model] + [B*n_vars, 1, d_model] -> [B*n_vars, patch_num+1, d_model]
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        # (4) 前馈网络模块：使用1D卷积实现
        # 转置以适应Conv1d: [B*n_vars, patch_num+1, d_model] -> [B*n_vars, d_model, patch_num+1]
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # 第二个卷积层并转置回来: [B*n_vars, d_ff, patch_num+1] -> [B*n_vars, d_model, patch_num+1] -> [B*n_vars, patch_num+1, d_model]
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # (5) 残差连接和最终层归一化
        return self.norm3(x + y)


class Model(nn.Module):
    """
    TimeXer模型 - 基于Transformer的时间序列预测模型
    
    TimeXer是一个专门为时间序列预测设计的Transformer变体模型。
    主要创新包括：
    1. 时间序列分块(patching)机制
    2. 双重嵌入结构：内生嵌入(EnEmbedding) + 外生嵌入(DataEmbedding_inverted)
    3. 交叉注意力机制：内生特征与外生特征的交互
    4. 编码器-预测头架构：无需解码器的高效设计
    
    理论依据:
    - Vision Transformer的patch embedding思想应用于时间序列
    - 内生-外生特征分离处理提高模型表达能力
    - Non-stationary Transformer的归一化技术
    
    支持任务:
    - 长期预测 (long_term_forecast)
    - 短期预测 (short_term_forecast)
    - 支持单变量(MS)和多变量(M)预测
    
    Args:
        configs: 配置对象，包含以下属性:
            - task_name (str): 任务类型
            - features (str): 特征类型，'M'表示多变量，'MS'表示单变量
            - seq_len (int): 输入序列长度
            - pred_len (int): 预测长度
            - use_norm (bool): 是否使用归一化
            - patch_len (int): patch长度
            - enc_in (int): 编码器输入特征数
            - d_model (int): 模型嵌入维度
            - dropout (float): dropout概率
            - embed (str): 嵌入类型
            - freq (str): 时间频率
            - factor (int): 注意力因子
            - n_heads (int): 注意力头数
            - d_ff (int): 前馈网络维度
            - e_layers (int): 编码器层数
            - activation (str): 激活函数类型
    
    用法示例:
        ```python
        import torch
        from argparse import Namespace
        
        # 创建配置
        configs = Namespace(
            task_name='long_term_forecast',
            features='M',
            seq_len=96,
            pred_len=24,
            use_norm=True,
            patch_len=16,
            enc_in=7,
            d_model=512,
            dropout=0.1,
            embed='timeF',
            freq='h',
            factor=1,
            n_heads=8,
            d_ff=2048,
            e_layers=2,
            activation='gelu'
        )
        
        # 创建模型
        model = Model(configs)
        
        # 准备输入数据
        batch_size = 32
        x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
        x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)  # 时间特征
        x_dec = torch.randn(batch_size, configs.pred_len, configs.enc_in)
        x_mark_dec = torch.randn(batch_size, configs.pred_len, 4)
        
        # 前向传播
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"输出形状: {output.shape}")  # [32, 24, 7]
        ```
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # 保存关键配置参数
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        # 计算patch数量
        self.patch_num = int(configs.seq_len // configs.patch_len)
        # 根据特征类型确定变量数量：MS(单变量)为1，M(多变量)为enc_in
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        
        # (1) 内生嵌入层：处理时间序列的主要特征，进行分块嵌入
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

        # (2) 外生嵌入层：处理时间标记等辅助特征，使用倒置嵌入
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        # (3) 编码器：多层Transformer编码器堆叠
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # 自注意力层：处理patch序列内部依赖
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    # 交叉注意力层：内生特征与外生特征交互
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)  # 创建e_layers个编码器层
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)  # 最终层归一化
        )
        
        # (4) 预测头部：计算输入特征数量并创建预测头
        # head_nf = d_model * (patch_num + 1)，+1是因为有全局token
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        单变量预测函数
        
        专门处理单变量时间序列预测任务（features='MS'或其他非'M'情况）。
        在单变量模式下，只使用时间序列的最后一个特征进行预测。
        
        Args:
            x_enc (torch.Tensor): 编码器输入序列
                形状: [batch_size, seq_len, features]
            x_mark_enc (torch.Tensor): 编码器时间标记
                形状: [batch_size, seq_len, mark_features]
            x_dec (torch.Tensor): 解码器输入序列（未使用）
            x_mark_dec (torch.Tensor): 解码器时间标记（未使用）
        
        Returns:
            torch.Tensor: 预测结果
                形状: [batch_size, pred_len, 1]
                
        张量形状变换流程:
            输入: [B, L, N]
            -> 取最后一维: [B, L, 1] -> permute -> [B, 1, L]
            -> en_embedding -> [B*1, patch_num+1, d_model]
            -> 取前N-1维: [B, L, N-1] -> ex_embedding -> [B, (N-1) + mark_features, d_model]
            -> encoder -> [B*1, patch_num+1, d_model]
            -> reshape+permute -> [B, 1, d_model, patch_num+1]
            -> head -> [B, 1, pred_len] -> permute -> [B, pred_len, 1]
        """
        # (1) 可选的输入归一化（Non-stationary Transformer技术）
        if self.use_norm:
            # 计算均值和标准差用于归一化
            means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, N]
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, N]
            x_enc /= stdev

        _, _, N = x_enc.shape

        # (2) 内生嵌入：仅使用最后一个特征维度
        # 取最后一维并调整维度：[B, L, N] -> [B, L, 1] -> [B, 1, L]
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        
        # (3) 外生嵌入：使用前N-1个特征维度和时间标记
        # [B, L, N-1] + [B, L, mark_features] -> [B, (N-1) + mark_features, d_model]
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        # (4) 编码器处理
        # en_embed: [B*1, patch_num+1, d_model], ex_embed: [B, (N-1) + mark_features, d_model]
        enc_out = self.encoder(en_embed, ex_embed)
        
        # (5) 重塑为四维张量用于预测头
        # [B*1, patch_num+1, d_model] -> [B, 1, patch_num+1, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # 调整维度顺序：[B, 1, patch_num+1, d_model] -> [B, 1, d_model, patch_num+1]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # (6) 预测头生成最终预测
        # [B, 1, d_model, patch_num+1] -> [B, 1, pred_len]
        dec_out = self.head(enc_out)
        # 调整输出维度：[B, 1, pred_len] -> [B, pred_len, 1]
        dec_out = dec_out.permute(0, 2, 1)

        # (7) 可选的反归一化
        if self.use_norm:
            # 使用最后一个特征的统计量进行反归一化
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        多变量预测函数
        
        专门处理多变量时间序列预测任务（features='M'）。
        在多变量模式下，所有特征维度都参与预测过程。
        
        Args:
            x_enc (torch.Tensor): 编码器输入序列
                形状: [batch_size, seq_len, features]
            x_mark_enc (torch.Tensor): 编码器时间标记
                形状: [batch_size, seq_len, mark_features]
            x_dec (torch.Tensor): 解码器输入序列（未使用）
            x_mark_dec (torch.Tensor): 解码器时间标记（未使用）
        
        Returns:
            torch.Tensor: 预测结果
                形状: [batch_size, pred_len, features]
                
        张量形状变换流程:
            输入: [B, L, N]
            -> permute -> [B, N, L]
            -> en_embedding -> [B*N, patch_num+1, d_model]
            -> ex_embedding: [B, L, N] -> [B, N + mark_features, d_model]
            -> encoder -> [B*N, patch_num+1, d_model]
            -> reshape+permute -> [B, N, d_model, patch_num+1]
            -> head -> [B, N, pred_len] -> permute -> [B, pred_len, N]
        """
        # (1) 可选的输入归一化
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, N]
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, N]
            x_enc /= stdev

        _, _, N = x_enc.shape

        # (2) 内生嵌入：使用所有特征维度
        # 调整维度：[B, L, N] -> [B, N, L]
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        
        # (3) 外生嵌入：使用完整输入和时间标记
        # [B, L, N] + [B, L, mark_features] -> [B, N + mark_features, d_model]
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        # (4) 编码器处理
        # en_embed: [B*N, patch_num+1, d_model], ex_embed: [B, N + mark_features, d_model]
        enc_out = self.encoder(en_embed, ex_embed)
        
        # (5) 重塑为四维张量
        # [B*N, patch_num+1, d_model] -> [B, N, patch_num+1, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # 调整维度顺序：[B, N, patch_num+1, d_model] -> [B, N, d_model, patch_num+1]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # (6) 预测头生成最终预测
        # [B, N, d_model, patch_num+1] -> [B, N, pred_len]
        dec_out = self.head(enc_out)
        # 调整输出维度：[B, N, pred_len] -> [B, pred_len, N]
        dec_out = dec_out.permute(0, 2, 1)

        # (7) 可选的反归一化
        if self.use_norm:
            # 使用所有特征的统计量进行反归一化
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        模型主要前向传播函数
        
        根据任务类型和特征类型，调用相应的预测函数。
        
        Args:
            x_enc (torch.Tensor): 编码器输入序列
                形状: [batch_size, seq_len, features]
            x_mark_enc (torch.Tensor): 编码器时间标记
                形状: [batch_size, seq_len, mark_features]
            x_dec (torch.Tensor): 解码器输入序列
                形状: [batch_size, label_len+pred_len, features]
            x_mark_dec (torch.Tensor): 解码器时间标记
                形状: [batch_size, label_len+pred_len, mark_features]
            mask (torch.Tensor, optional): 掩码张量（当前未使用）
        
        Returns:
            torch.Tensor: 预测结果
                - 多变量预测: [batch_size, pred_len, features]
                - 单变量预测: [batch_size, pred_len, 1]
                - 不支持的任务: None
        """
        # 长期预测或短期预测任务
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                # 多变量预测
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            else:
                # 单变量预测（features='MS'或其他）
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            # 不支持的任务类型
            return None