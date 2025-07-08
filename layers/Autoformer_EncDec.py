"""
Autoformer Encoder-Decoder 相关模块

本文件实现了 Autoformer/FEDformer 论文中的若干核心组件，用于时间序列建模与分解：
    • `moving_avg`               — 计算滑动平均以提取趋势 (Trend)。
    • `series_decomp`            — 将序列分解为季节性 (Seasonal) 与趋势 (Trend)。
    • `series_decomp_multi`      — 多尺度分解，可并行使用多种窗口。
    • `EncoderLayer` / `Encoder` — 逐层自注意力 + 分解的编码架构。
    • `DecoderLayer` / `Decoder` — 解码端采用交叉注意力与递进分解。

该模块既可供 Autoformer 使用，也被 DLinear 等模型复用。

张量记号约定：
    B — batch_size
    L — 序列长度 (seq_len)
    D — 变量维度 (d_model / channels)

所有方法均严格注明输入 / 输出形状，方便新手理解与调试。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """专用于季节性分量的 LayerNorm。

    与标准 `nn.LayerNorm` 不同，本实现会在归一化后去除通道均值 (bias)，
    以保证季节性分量均值为 0，符合论文中的设计。

    Args:
        channels (int): 输入特征维度 \(D)。
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        """前向计算。

        Args:
            x (torch.Tensor): 输入张量，形状 `(B, L, D)`。

        Returns:
            torch.Tensor: 去均值后的归一化结果，形状 `(B, L, D)`。
        """
        # (1) 常规 LayerNorm：保持形状不变
        x_hat = self.layernorm(x)
        # (2) 计算 batch 内每个时间步均值，并复制到原形状 (B, 1, D) → (B, L, D)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        # (3) 去除均值，使季节性分量零均值
        return x_hat - bias


class moving_avg(nn.Module):
    """滑动平均模块，用于提取时间序列趋势 (Trend)。

    Args:
        kernel_size (int): 卷积窗口大小，相当于滑动窗口长度。
        stride (int): 步幅，默认为 1。

    输入形状: `(B, L, D)`
    输出形状: `(B, L, D)` （与输入一致）。
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """前向计算滑动平均。

        形状变换流程:
            (B, L, D)
                ➡️ 前后端填充 → (B, L + k - 1, D)
                ➡️ 转置后 1D AvgPool → (B, D, L)
                ➡️ 再转置回来 → (B, L, D)
        """
        # (1) padding — 在序列两端复制首尾元素，避免边界信息缺失
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # (2) 转换为 (B, D, L) 以沿时间维做 AvgPool
        x = self.avg(x.permute(0, 2, 1))
        # (3) 恢复原始形状 (B, L, D)
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """单尺度时间序列分解模块。

    使用滑动平均将序列 \(x) 分解为:
        • `res`: 季节性 / 残差分量，`x - moving_mean`
        • `moving_mean`: 趋势分量

    Args:
        kernel_size (int): 滑动平均窗口大小。

    输入形状: `(B, L, D)`
    输出形状: `(res, trend)` — 均为 `(B, L, D)`。
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """执行序列分解。

        Steps:
            (1) 计算滑动平均 → `moving_mean` (Trend)
            (2) 计算残差      → `res = x - moving_mean` (Seasonal)
        """
        # (1) 滑动平均，提取趋势
        moving_mean = self.moving_avg(x)
        # (2) 残差 = 原序列 - 趋势
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """多尺度序列分解模块 (来源: FEDformer)。

    接收一个窗口列表 `kernel_size`，为每个窗口实例化一个 `series_decomp`，
    最终输出各尺度平均后的季节性与趋势分量。
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
