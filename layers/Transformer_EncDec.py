import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """一维卷积层
    用于对输入序列进行下采样和特征提取
    """
    def __init__(self, c_in):
        """
        参数:
            c_in: 输入通道数
        """
        super(ConvLayer, self).__init__()
        # 一维卷积层，保持通道数不变，kernel_size=3
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')  # 使用循环填充模式
        self.norm = nn.BatchNorm1d(c_in)  # 批归一化层
        self.activation = nn.ELU()  # 激活函数
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 最大池化层，步长为2，实现下采样

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状为 [batch_size, seq_len, channels]
        返回:
            下采样后的特征图，形状为 [batch_size, seq_len/2, channels]
        """
        x = self.downConv(x.permute(0, 2, 1))  # 调整维度顺序并进行卷积
        x = self.norm(x)  # 批归一化
        x = self.activation(x)  # 激活函数
        x = self.maxPool(x)  # 最大池化
        x = x.transpose(1, 2)  # 恢复维度顺序
        return x


class EncoderLayer(nn.Module):
    """Transformer编码器层
    包含自注意力机制和前馈神经网络
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        参数:
            attention: 注意力机制模块
            d_model: 模型维度
            d_ff: 前馈网络维度，默认为4*d_model
            dropout: dropout比率
            activation: 激活函数类型
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # 两个一维卷积实现前馈网络
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二个层归一化
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        前向传播
        参数:
            x: 输入张量
            attn_mask: 注意力掩码
            tau: 温度参数
            delta: 相对位置信息
        返回:
            处理后的特征和注意力权重
        """
        # 自注意力机制
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)  # 残差连接

        y = x = self.norm1(x)
        # 前馈网络部分
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn  # 残差连接和层归一化


class Encoder(nn.Module):
    """Transformer编码器
    由多个编码器层和可选的卷积层组成
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        参数:
            attn_layers: 注意力层列表
            conv_layers: 卷积层列表（可选）
            norm_layer: 归一化层（可选）
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        前向传播
        参数:
            x: 输入张量 [batch_size, seq_len, dim]
            attn_mask: 注意力掩码
            tau: 温度参数
            delta: 相对位置信息
        返回:
            处理后的特征和所有层的注意力权重
        """
        attns = []
        if self.conv_layers is not None:
            # 交替使用注意力层和卷积层
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            # 只使用注意力层
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """Transformer解码器层
    包含自注意力、交叉注意力和前馈神经网络
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        """
        参数:
            self_attention: 自注意力模块
            cross_attention: 交叉注意力模块
            d_model: 模型维度
            d_ff: 前馈网络维度
            dropout: dropout比率
            activation: 激活函数类型
        """
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        # 前馈网络
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # 三个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        前向传播
        参数:
            x: 输入张量
            cross: 编码器输出的特征
            x_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
            tau: 温度参数
            delta: 相对位置信息
        返回:
            处理后的特征
        """
        # 自注意力
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # 交叉注意力
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        # 前馈网络
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    """Transformer解码器
    由多个解码器层组成
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        """
        参数:
            layers: 解码器层列表
            norm_layer: 归一化层（可选）
            projection: 投影层（可选）
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        前向传播
        参数:
            x: 输入张量
            cross: 编码器输出的特征
            x_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
            tau: 温度参数
            delta: 相对位置信息
        返回:
            处理后的特征
        """
        # 依次通过所有解码器层
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
