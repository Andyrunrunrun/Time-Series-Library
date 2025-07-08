import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # freq_k = 1 / (10000^(2k/d_model))
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 在张量的最前面添加一个维度 (max_len, d_model) -> (1, max_len, d_model)
        self.pe: torch.Tensor = pe  # 添加类型注释
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]  # 获取输入张量 x 在第二个维度（索引1）的大小，也就是序列长度


class TokenEmbedding(nn.Module):
    """时间序列数据的令牌嵌入层
    
    将输入的时间序列数据通过一维卷积转换为具有固定维度的嵌入表示。
    使用循环填充（circular padding）来处理序列的边界，保持时间连续性。
    
    参数:
        c_in (int): 输入特征维度
        d_model (int): 输出嵌入维度
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # 根据PyTorch版本选择不同的填充值
        padding = 1 if torch.version.__version__ >= "1.5.0" else 2
        
        # 定义一维卷积层
        self.tokenConv = nn.Conv1d( # PyTorch的 Conv1d 期望的输入格式是 [batch_size, channels, seq_length]
            in_channels=c_in,      # 输入通道数，即特征维度
            out_channels=d_model,  # 输出通道数，即嵌入维度
            kernel_size=3,         # 卷积核大小为3，可以捕获局部时间依赖
            padding=padding,       # 填充大小，保持序列长度不变
            padding_mode="circular", # 循环填充，处理时间序列边界
            bias=False,            # 不使用偏置项
        )
        
        # 使用Kaiming初始化方法初始化卷积层权重
        # 这种初始化方法特别适合使用ReLU及其变体作为激活函数的深度网络
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_in",         # 使用fan_in模式，考虑输入维度进行归一化
                    nonlinearity="leaky_relu" # 假设使用leaky_relu激活函数
                )

    def forward(self, x):
        """前向传播函数
        
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, seq_length, channels]
            
        返回:
            Tensor: 输出张量，形状为 [batch_size, seq_length, d_model]
        """
        # 1. permute(0,2,1): 将输入张量从[batch_size, seq_length, channels]转换为[batch_size, channels, seq_length]
        # 2. 应用卷积操作
        # 3. transpose(1,2): 将结果转换回[batch_size, seq_length, d_model]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """固定嵌入层
    
    使用正弦和余弦函数实现的固定编码方式，不需要训练。
    这种编码方式的优点是：
    1. 可以处理任意长度的序列
    2. 编码值是确定的，不需要学习
    3. 不同位置的编码之间存在可预测的线性关系
    
    参数:
        c_in (int): 输入维度，即需要编码的类别数量
        d_model (int): 输出的嵌入维度，必须是偶数（因为要平均分配给sin和cos）
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # 创建一个不需要梯度的空张量来存储编码
        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        # 生成位置编码
        position = torch.arange(0, c_in).float().unsqueeze(1)  # 形状: [c_in, 1]
        
        # 计算不同维度的频率因子
        # 使用指数递减的频率：1/10000^(2i/d_model)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        # 使用正弦和余弦函数生成编码
        # 偶数维度使用正弦，奇数维度使用余弦
        w[:, 0::2] = torch.sin(position * div_term)  # sin(pos * freq)
        w[:, 1::2] = torch.cos(position * div_term)  # cos(pos * freq)

        # 创建一个嵌入层，但使用我们预先计算好的权重
        self.emb = nn.Embedding(c_in, d_model)
        # 将预计算的权重设置为嵌入层的权重，并设置为不可训练
        self.emb.weight = Parameter(w, requires_grad=False)

    def forward(self, x):
        """前向传播函数
        
        参数:
            x (Tensor): 输入的索引张量，形状为 [batch_size, seq_length]
                       值应该在范围 [0, c_in-1] 内
            
        返回:
            Tensor: 固定的嵌入向量，形状为 [batch_size, seq_length, d_model]
                   使用detach()确保不会计算梯度
        """
        return self.emb(x).detach()  # detach确保不会计算梯度


class TemporalEmbedding(nn.Module):
    """时间特征嵌入层
    
    将时间特征（月份、星期几、天、小时、分钟）转换为向量表示。
    支持两种嵌入方式：
    1. fixed: 使用固定的正弦余弦编码（类似Transformer的位置编码）
    2. learned: 使用可学习的标准嵌入层
    
    参数:
        d_model (int): 嵌入向量的维度
        embed_type (str): 嵌入类型，可选 'fixed' 或 'learned'
        freq (str): 时间序列的频率，例如 'h'表示小时级别，'t'表示分钟级别
    """
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        # 定义各个时间尺度的大小
        minute_size = 4    # 每小时分为4个15分钟段
        hour_size = 24     # 24小时
        weekday_size = 7   # 一周7天
        day_size = 32      # 每月最多31天（用32是为了包含0）
        month_size = 13    # 12个月（用13是为了包含0）

        # 根据embed_type选择使用固定编码还是可学习的嵌入层
        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        
        # 根据时间序列的频率freq决定需要哪些时间特征的嵌入
        if freq == "t":  # 如果是分钟级别的数据，需要分钟嵌入
            self.minute_embed = Embed(minute_size, d_model)
            
        # 创建各个时间尺度的嵌入层
        self.hour_embed = Embed(hour_size, d_model)       # 小时嵌入
        self.weekday_embed = Embed(weekday_size, d_model) # 星期嵌入
        self.day_embed = Embed(day_size, d_model)         # 天嵌入
        self.month_embed = Embed(month_size, d_model)     # 月份嵌入

    def forward(self, x):
        """前向传播函数
        
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, seq_length, 5]
                       最后一维包含5个时间特征：[月, 日, 星期几, 小时, 分钟]
                       
        返回:
            Tensor: 所有时间特征的嵌入向量之和，形状为 [batch_size, seq_length, d_model]
        """
        x = x.long()  # 将输入转换为长整型，因为这些是离散的类别索引
        
        # 获取各个时间特征的嵌入向量
        # x[:, :, 4]表示第4个特征（分钟），如果有分钟嵌入层则使用，否则返回0
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        
        # 分别获取小时、星期几、日期、月份的嵌入向量
        hour_x = self.hour_embed(x[:, :, 3])         # 第3个特征是小时
        weekday_x = self.weekday_embed(x[:, :, 2])   # 第2个特征是星期几
        day_x = self.day_embed(x[:, :, 1])           # 第1个特征是日期
        month_x = self.month_embed(x[:, :, 0])       # 第0个特征是月份

        # 将所有时间特征的嵌入向量相加，得到最终的时间嵌入表示
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """Patch Embedding 层

    该层首先按照 `patch_len` 与 `stride` 将时间序列切分成多个 patch（即时间片段），
    对每个 patch 进行线性投影得到固定维度的表示，随后加上可学习的位置编码以提供
    时序位置信息，并最终施加 dropout。

    参数说明:
        d_model (int): 线性投影后的特征维度 (embedding 维度)
        patch_len (int): 每个 patch 的长度 (窗口大小)
        stride (int): 相邻 patch 之间的步幅，stride < patch_len 时 patch 会重叠
        padding (int): 在序列右侧复制填充的长度，用于保证序列可以被整除地切分
        dropout (float): 位置编码之后施加的 dropout 比例
    输入形状:
        x: Tensor，形状为 [batch_size, n_vars, seq_len]
    输出:
        Tuple(Tensor, int):
            - patch 表示张量，形状为 [batch_size * n_vars, n_patches, d_model]
            - n_vars: 原始变量 (通道) 数，用于后续模型还原批量维度
    """

    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float):
        super(PatchEmbedding, self).__init__()

        # 保存 patch 超参，便于 forward 使用
        self.patch_len = patch_len  # 每个 patch 的时间步数
        self.stride = stride        # 相邻 patch 起始位置的步幅

        # 复制填充 (replication padding) —— 将序列右侧复制 `padding` 个时间步，
        # 以确保最后一个 patch 的长度充足。
        # nn.ReplicationPad1d 期望输入的维度顺序为 [B, C, L]，但我们稍后会保证满足这一点。
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # 对展开后的 patch 做线性投影，得到 d_model 维度的表示。
        # 注意 input_features 为 patch_len，因为我们把时间维当作特征维输入 Linear。
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # 位置编码，用于保留 patch 在整体序列中的位置信息
        self.position_embedding = PositionalEmbedding(d_model)

        # dropout，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播

        步骤:
            1. 记录变量维度 `n_vars`，稍后展开后需要用到。
            2. 复制填充以确保切分窗口完整。
            3. 调用 `unfold` 在最后一个维度 (时间维) 上做滑动窗口，生成形状
               [batch_size, n_vars, n_patches, patch_len] 的张量。
            4. 将 batch 维与变量维合并，得到形状 [batch_size * n_vars, n_patches, patch_len]，
               这样后续 Transformer 可以把每个变量视作独立的序列处理。
            5. 对 patch 做线性投影并加上位置编码。
            6. 返回 dropout 处理后的结果以及变量数 `n_vars`。
        """

        # 1. 记录变量（通道）数
        n_vars = x.shape[1]

        # 2. 复制填充，保持长度整除关系；此时 x 仍为 [B, C, L]
        x = self.padding_patch_layer(x)

        # 3. unfold: 在时间维 (dim=-1) 上切分 patch
        #    输出形状: [B, C, n_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # 4. 将 (batch, channel) 两个维度压平成一个维度，以便后续作为序列批处理
        #    新形状: [B * C, n_patches, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # 5. patch 线性投影 + 位置编码
        x = self.value_embedding(x) + self.position_embedding(x)

        # 6. dropout 并返回结果以及通道数，供后续还原形状使用
        return self.dropout(x), n_vars
