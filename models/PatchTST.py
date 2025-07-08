# 导入必要的深度学习库和模块
import torch  # PyTorch深度学习框架
from torch import nn  # PyTorch神经网络模块

# 导入自定义的Transformer编码器相关组件
from layers.Transformer_EncDec import Encoder, EncoderLayer  # Transformer编码器和编码器层
from layers.SelfAttention_Family import FullAttention, AttentionLayer  # 自注意力机制相关层
from layers.Embed import PatchEmbedding  # 补丁嵌入层，用于将时间序列数据转换为补丁表示

class Transpose(nn.Module):
    """
    自定义的转置操作层
    
    这个模块用于在神经网络中进行张量维度的转置操作，
    特别适用于需要改变张量维度顺序的场景。
    
    参数:
        *dims: 要转置的维度索引
        contiguous: 是否确保输出张量在内存中是连续的
    """
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入张量
            
        返回:
            转置后的张量
        """
        if self.contiguous: 
            return x.transpose(*self.dims).contiguous()  # 转置并确保内存连续性
        else: 
            return x.transpose(*self.dims)  # 仅进行转置操作


class FlattenHead(nn.Module):
    """
    预测头部网络，用于将编码器输出转换为最终预测结果
    
    这个模块是PatchTST模型的最后一层，负责将Transformer编码器
    的输出特征转换为时间序列预测的目标长度。
    
    参数:
        n_vars: 变量数量（时间序列的特征维度）
        nf: 输入特征数量（d_model * patch_num）
        target_window: 目标输出长度（预测长度或序列长度）
        head_dropout: dropout概率，用于防止过拟合
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars  # 保存变量数量
        # 展平操作：将最后两个维度展平为一个维度
        self.flatten = nn.Flatten(start_dim=-2)
        # 线性层：将特征维度映射到目标输出长度
        self.linear = nn.Linear(nf, target_window)
        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        """
        前向传播函数
        
        参数:
            x: 输入张量，形状为 [bs x nvars x d_model x patch_num]
               其中 bs=批次大小, nvars=变量数量, d_model=模型维度, patch_num=补丁数量
               
        返回:
            输出张量，形状为 [bs x nvars x target_window]
        """
        # 展平最后两个维度：[bs x nvars x d_model x patch_num] -> [bs x nvars x (d_model * patch_num)]
        x = self.flatten(x)
        # 线性变换：[bs x nvars x (d_model * patch_num)] -> [bs x nvars x target_window]
        x = self.linear(x)
        # 应用dropout进行正则化
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    PatchTST模型 - 基于Transformer的时间序列预测模型
    
    论文链接: https://arxiv.org/pdf/2211.14730.pdf
    
    PatchTST（Patched Time Series Transformer）是一种专门用于时间序列分析的Transformer架构。
    该模型通过将时间序列数据分割成补丁（patch），然后使用自注意力机制进行建模，
    能够有效处理长序列预测、缺失值插补、异常检测和分类等多种任务。
    
    模型主要特点：
    1. 补丁嵌入：将时间序列分割成固定长度的补丁
    2. 自注意力机制：捕获序列中的长期依赖关系
    3. 多任务支持：支持预测、插补、异常检测和分类任务
    4. 标准化处理：使用Non-stationary Transformer的标准化技术
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        PatchTST模型初始化函数
        
        参数:
            configs: 配置对象，包含模型的各种超参数
                - task_name: 任务类型（'long_term_forecast', 'short_term_forecast', 'imputation', 'anomaly_detection', 'classification'）
                - seq_len: 输入序列长度
                - pred_len: 预测序列长度
                - d_model: 模型隐藏层维度
                - n_heads: 注意力头数
                - e_layers: 编码器层数
                - d_ff: 前馈网络维度
                - dropout: dropout概率
                - activation: 激活函数类型
                - factor: 注意力机制的缩放因子
                - enc_in: 输入特征维度
                - num_class: 分类任务的类别数
            patch_len: 补丁长度，将时间序列分割成指定长度的补丁，默认16
            stride: 滑动窗口步长，控制补丁之间的重叠程度，默认8
        """
        super().__init__()
        
        # 保存任务相关的配置信息
        self.task_name = configs.task_name  # 任务类型
        self.seq_len = configs.seq_len      # 输入序列长度
        self.pred_len = configs.pred_len    # 预测序列长度
        
        # 设置补丁填充策略，这里使用与步长相同的填充
        padding = stride

        # 1. 补丁嵌入层 - 将时间序列数据转换为补丁表示
        # 这是PatchTST的核心创新，将连续的时间序列分割成重叠的补丁
        self.patch_embedding = PatchEmbedding(
            configs.d_model,    # 输出特征维度
            patch_len,          # 补丁长度
            stride,             # 滑动步长
            padding,            # 填充大小
            configs.dropout     # dropout概率
        )

        # 2. Transformer编码器 - 使用自注意力机制处理补丁序列
        # 编码器由多个编码器层组成，每层包含多头自注意力和前馈网络
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # 注意力层配置
                    AttentionLayer(
                        FullAttention(
                            False,                           # 不使用掩码
                            configs.factor,                  # 注意力缩放因子
                            attention_dropout=configs.dropout,  # 注意力dropout
                            output_attention=False           # 不输出注意力权重
                        ), 
                        configs.d_model,     # 模型维度
                        configs.n_heads      # 注意力头数
                    ),
                    configs.d_model,         # 模型维度
                    configs.d_ff,            # 前馈网络维度
                    dropout=configs.dropout, # dropout概率
                    activation=configs.activation  # 激活函数
                ) for _ in range(configs.e_layers)  # 创建指定数量的编码器层
            ],
            # 标准化层：先转置 -> 批归一化 -> 再转置
            # 这种设计是为了适应批归一化对特征维度的要求
            norm_layer=nn.Sequential(
                Transpose(1,2),                          # 转置：[batch, seq, features] -> [batch, features, seq]
                nn.BatchNorm1d(configs.d_model),         # 批归一化
                Transpose(1,2)                           # 转置回来：[batch, features, seq] -> [batch, seq, features]
            )
        )

        # 3. 预测头部网络 - 将编码器输出转换为最终预测结果
        # 计算头部网络的输入特征数量
        # 公式：(seq_len - patch_len) / stride + 2 计算补丁数量
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        
        # 根据不同任务类型配置不同的预测头
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 预测任务：输出长度为pred_len
            self.head = FlattenHead(
                configs.enc_in,      # 输入变量数量
                self.head_nf,        # 特征数量
                configs.pred_len,    # 预测长度
                head_dropout=configs.dropout  # 头部dropout
            )
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            # 插补和异常检测任务：输出长度与输入序列长度相同
            self.head = FlattenHead(
                configs.enc_in,      # 输入变量数量
                self.head_nf,        # 特征数量
                configs.seq_len,     # 序列长度
                head_dropout=configs.dropout  # 头部dropout
            )
        elif self.task_name == 'classification':
            # 分类任务：使用不同的头部结构
            self.flatten = nn.Flatten(start_dim=-2)  # 展平特征
            self.dropout = nn.Dropout(configs.dropout)  # dropout层
            # 线性投影层：将特征映射到类别数量
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in,  # 输入特征数量
                configs.num_class               # 输出类别数量
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        时间序列预测方法
        
        这是PatchTST模型的核心预测流程，包含以下主要步骤：
        1. 数据标准化（Non-stationary Transformer方法）
        2. 补丁嵌入和特征提取
        3. Transformer编码器处理
        4. 预测头部生成结果
        5. 反标准化恢复原始尺度
        
        参数:
            x_enc: 编码器输入，形状为 [batch_size, seq_len, features]
            x_mark_enc: 编码器时间标记（未使用，保留接口兼容性）
            x_dec: 解码器输入（未使用，保留接口兼容性）
            x_mark_dec: 解码器时间标记（未使用，保留接口兼容性）
            
        返回:
            dec_out: 预测结果，形状为 [batch_size, pred_len, features]
        """
        # === 第1步：数据标准化 ===
        # 使用Non-stationary Transformer的标准化方法，提高模型对非平稳时间序列的处理能力
        # 计算每个序列的均值，用于后续的去均值操作
        means = x_enc.mean(1, keepdim=True).detach()  # [batch_size, 1, features]
        # 去均值：减去序列均值，使数据围绕零点分布
        x_enc = x_enc - means
        # 计算标准差，用于标准化，添加小常数防止除零
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [batch_size, 1, features]
        # 标准化：除以标准差，使数据方差为1
        x_enc /= stdev

        # === 第2步：补丁嵌入和特征提取 ===
        # 转换维度：[batch_size, seq_len, features] -> [batch_size, features, seq_len]
        # 这种转换是为了适应补丁嵌入层的输入格式要求
        x_enc = x_enc.permute(0, 2, 1)
        # 应用补丁嵌入：将时间序列切分成补丁并嵌入到高维空间
        # 输出形状：[batch_size * features, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # === 第3步：Transformer编码器处理 ===
        # 使用自注意力机制处理补丁序列，捕获长期依赖关系
        # 输入输出形状：[batch_size * features, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out)
        
        # === 第4步：重塑张量为预测头部输入格式 ===
        # 恢复批次和变量维度：[batch_size * features, patch_num, d_model] -> [batch_size, features, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # 调整维度顺序：[batch_size, features, patch_num, d_model] -> [batch_size, features, d_model, patch_num]
        # 这种转换是为了适应预测头部的输入格式
        enc_out = enc_out.permute(0, 1, 3, 2)

        # === 第5步：预测头部生成结果 ===
        # 应用预测头部，生成预测结果
        # 输入形状：[batch_size, features, d_model, patch_num]
        # 输出形状：[batch_size, features, pred_len]
        dec_out = self.head(enc_out)
        # 调整维度顺序：[batch_size, features, pred_len] -> [batch_size, pred_len, features]
        dec_out = dec_out.permute(0, 2, 1)

        # === 第6步：反标准化恢复原始尺度 ===
        # 恢复标准差缩放：乘以原始标准差
        # 需要调整标准差张量的形状以匹配预测结果
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # 恢复均值偏移：加上原始均值
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        时间序列缺失值插补方法
        
        该方法用于处理时间序列中的缺失值，通过掩码机制识别缺失位置，
        并使用PatchTST模型学习到的表示来预测缺失值。
        
        主要步骤：
        1. 基于掩码的标准化处理
        2. 补丁嵌入和特征提取
        3. Transformer编码器处理
        4. 预测头部生成完整序列
        5. 反标准化恢复原始尺度
        
        参数:
            x_enc: 编码器输入，形状为 [batch_size, seq_len, features]，包含缺失值
            x_mark_enc: 编码器时间标记（未使用，保留接口兼容性）
            x_dec: 解码器输入（未使用，保留接口兼容性）
            x_mark_dec: 解码器时间标记（未使用，保留接口兼容性）
            mask: 掩码矩阵，形状为 [batch_size, seq_len, features]，
                  1表示有效值，0表示缺失值
                  
        返回:
            dec_out: 插补后的完整序列，形状为 [batch_size, seq_len, features]
        """
        # === 第1步：基于掩码的数据标准化 ===
        # 插补任务需要特殊的标准化处理，因为存在缺失值
        # 计算有效值的均值（排除缺失值）
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)  # [batch_size, features]
        means = means.unsqueeze(1).detach()  # [batch_size, 1, features]
        
        # 去均值：减去有效值的均值
        x_enc = x_enc - means
        # 将缺失值位置填充为0，避免缺失值影响后续计算
        x_enc = x_enc.masked_fill(mask == 0, 0)
        
        # 计算有效值的标准差（基于掩码）
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()  # [batch_size, 1, features]
        # 标准化：除以标准差
        x_enc /= stdev

        # === 第2步：补丁嵌入和特征提取 ===
        # 转换维度：[batch_size, seq_len, features] -> [batch_size, features, seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # 应用补丁嵌入：将时间序列切分成补丁并嵌入到高维空间
        # 输出形状：[batch_size * features, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # === 第3步：Transformer编码器处理 ===
        # 使用自注意力机制处理补丁序列，模型能够利用有效值的信息来推断缺失值
        # 输入输出形状：[batch_size * features, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out)
        
        # === 第4步：重塑张量为预测头部输入格式 ===
        # 恢复批次和变量维度：[batch_size * features, patch_num, d_model] -> [batch_size, features, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # 调整维度顺序：[batch_size, features, patch_num, d_model] -> [batch_size, features, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # === 第5步：预测头部生成完整序列 ===
        # 应用预测头部，生成完整的序列（包括插补的缺失值）
        # 输入形状：[batch_size, features, d_model, patch_num]
        # 输出形状：[batch_size, features, seq_len]
        dec_out = self.head(enc_out)
        # 调整维度顺序：[batch_size, features, seq_len] -> [batch_size, seq_len, features]
        dec_out = dec_out.permute(0, 2, 1)

        # === 第6步：反标准化恢复原始尺度 ===
        # 恢复标准差缩放：乘以原始标准差
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        # 恢复均值偏移：加上原始均值
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        时间序列异常检测方法
        
        该方法用于检测时间序列中的异常模式。通过重构输入序列，
        比较原始序列与重构序列的差异来识别异常点。
        异常点通常表现为重构误差较大的时间点。
        
        主要步骤：
        1. 数据标准化处理
        2. 补丁嵌入和特征提取
        3. Transformer编码器处理
        4. 预测头部重构序列
        5. 反标准化恢复原始尺度
        
        参数:
            x_enc: 编码器输入，形状为 [batch_size, seq_len, features]
            
        返回:
            dec_out: 重构后的序列，形状为 [batch_size, seq_len, features]
                    通过与原始序列比较可以计算异常分数
        """
        # === 第1步：数据标准化 ===
        # 使用Non-stationary Transformer的标准化方法
        # 计算每个序列的均值，用于去均值操作
        means = x_enc.mean(1, keepdim=True).detach()  # [batch_size, 1, features]
        # 去均值：减去序列均值
        x_enc = x_enc - means
        # 计算标准差，用于标准化
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [batch_size, 1, features]
        # 标准化：除以标准差
        x_enc /= stdev

        # === 第2步：补丁嵌入和特征提取 ===
        # 转换维度：[batch_size, seq_len, features] -> [batch_size, features, seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # 应用补丁嵌入：将时间序列切分成补丁并嵌入到高维空间
        # 输出形状：[batch_size * features, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # === 第3步：Transformer编码器处理 ===
        # 使用自注意力机制处理补丁序列，学习正常模式的表示
        # 异常模式由于偏离正常分布，在重构时会产生较大误差
        # 输入输出形状：[batch_size * features, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out)
        
        # === 第4步：重塑张量为预测头部输入格式 ===
        # 恢复批次和变量维度：[batch_size * features, patch_num, d_model] -> [batch_size, features, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # 调整维度顺序：[batch_size, features, patch_num, d_model] -> [batch_size, features, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # === 第5步：预测头部重构序列 ===
        # 应用预测头部，生成重构的序列
        # 正常数据应该能够被很好地重构，而异常数据重构误差较大
        # 输入形状：[batch_size, features, d_model, patch_num]
        # 输出形状：[batch_size, features, seq_len]
        dec_out = self.head(enc_out)
        # 调整维度顺序：[batch_size, features, seq_len] -> [batch_size, seq_len, features]
        dec_out = dec_out.permute(0, 2, 1)

        # === 第6步：反标准化恢复原始尺度 ===
        # 恢复标准差缩放：乘以原始标准差
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        # 恢复均值偏移：加上原始均值
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        时间序列分类方法
        
        该方法用于对时间序列进行分类，将整个时间序列序列映射到预定义的类别中。
        与其他任务不同，分类任务需要将序列级别的信息聚合为单一的类别预测。
        
        主要步骤：
        1. 数据标准化处理
        2. 补丁嵌入和特征提取
        3. Transformer编码器处理
        4. 全局特征聚合
        5. 分类头部输出类别概率
        
        参数:
            x_enc: 编码器输入，形状为 [batch_size, seq_len, features]
            x_mark_enc: 编码器时间标记（未使用，保留接口兼容性）
            
        返回:
            output: 分类logits，形状为 [batch_size, num_classes]
        """
        # === 第1步：数据标准化 ===
        # 使用Non-stationary Transformer的标准化方法
        # 计算每个序列的均值，用于去均值操作
        means = x_enc.mean(1, keepdim=True).detach()  # [batch_size, 1, features]
        # 去均值：减去序列均值
        x_enc = x_enc - means
        # 计算标准差，用于标准化
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [batch_size, 1, features]
        # 标准化：除以标准差
        x_enc /= stdev

        # === 第2步：补丁嵌入和特征提取 ===
        # 转换维度：[batch_size, seq_len, features] -> [batch_size, features, seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # 应用补丁嵌入：将时间序列切分成补丁并嵌入到高维空间
        # 输出形状：[batch_size * features, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # === 第3步：Transformer编码器处理 ===
        # 使用自注意力机制处理补丁序列，学习判别性特征表示
        # 编码器提取对分类任务有用的高级特征
        # 输入输出形状：[batch_size * features, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out)
        
        # === 第4步：重塑张量为分类头部输入格式 ===
        # 恢复批次和变量维度：[batch_size * features, patch_num, d_model] -> [batch_size, features, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # 调整维度顺序：[batch_size, features, patch_num, d_model] -> [batch_size, features, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # === 第5步：全局特征聚合和分类 ===
        # 展平所有特征维度：[batch_size, features, d_model, patch_num] -> [batch_size, features * d_model * patch_num]
        output = self.flatten(enc_out)
        # 应用dropout防止过拟合
        output = self.dropout(output)
        # 重塑为二维张量，确保与线性层兼容
        output = output.reshape(output.shape[0], -1)  # [batch_size, total_features]
        # 线性投影到类别空间：[batch_size, total_features] -> [batch_size, num_classes]
        output = self.projection(output)
        
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        PatchTST模型的前向传播函数
        
        这是模型的主要入口点，根据任务类型调用相应的处理方法。
        PatchTST支持多种时间序列任务，通过统一的接口提供不同的功能。
        
        参数:
            x_enc: 编码器输入，形状为 [batch_size, seq_len, features]
            x_mark_enc: 编码器时间标记（在某些任务中未使用）
            x_dec: 解码器输入（在某些任务中未使用）
            x_mark_dec: 解码器时间标记（在某些任务中未使用）
            mask: 掩码矩阵，仅在插补任务中使用，指示缺失值位置
            
        返回:
            根据任务类型返回不同形状的张量:
            - 预测任务: [batch_size, pred_len, features] - 预测的未来值
            - 插补任务: [batch_size, seq_len, features] - 插补后的完整序列
            - 异常检测: [batch_size, seq_len, features] - 重构后的序列
            - 分类任务: [batch_size, num_classes] - 类别概率logits
        """
        # === 长期和短期预测任务 ===
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 调用预测方法，生成未来时间步的预测值
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 只返回预测长度的部分，确保输出维度正确
            return dec_out[:, -self.pred_len:, :]  # [batch_size, pred_len, features]
        
        # === 缺失值插补任务 ===
        if self.task_name == 'imputation':
            # 调用插补方法，使用掩码信息重构完整序列
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [batch_size, seq_len, features]
        
        # === 异常检测任务 ===
        if self.task_name == 'anomaly_detection':
            # 调用异常检测方法，重构输入序列用于异常分数计算
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [batch_size, seq_len, features]
        
        # === 分类任务 ===
        if self.task_name == 'classification':
            # 调用分类方法，将序列映射到类别空间
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [batch_size, num_classes]
        
        # === 未知任务类型 ===
        # 如果任务类型不匹配，返回None（理论上不应该发生）
        return None
