import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    一个可配置的归一化模块，可以执行标准的Z-score归一化或基于最后一个时间步的减法。
    它还支持可学习的仿射变换（类似LayerNorm）以及跳过归一化。
    该模块可以在“norm”（归一化）和“denorm”（反归一化）两种模式下运行。
    """
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        初始化Normalize模块。
        :param num_features: int, 输入特征的数量或通道数。
        :param eps: float, 一个为了数值稳定性而加到分母（标准差）上的小值，防止除以零。
        :param affine: bool, 如果为True，则模块将拥有可学习的仿射参数（权重和偏置），用于在归一化后对数据进行缩放和平移。
        :param subtract_last: bool, 如果为True，则归一化将通过减去序列的最后一个时间步的值来完成，而不是减去均值。
        :param non_norm: bool, 如果为True，则完全跳过归一化和反归一化操作，模块将直接返回输入张量。
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        定义前向传播。
        根据 'mode' 参数执行归一化或反归一化。
        :param x: torch.Tensor, 输入张量，形状通常为 (batch_size, sequence_length, num_features)。
        :param mode: str, 操作模式，'norm' 表示归一化，'denorm' 表示反归一化。
        :return: torch.Tensor, 处理后的张量。
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            # 如果模式不是 'norm' 或 'denorm'，则抛出错误。
            raise NotImplementedError
        return x

    def _init_params(self):
        """
        如果 affine=True，则初始化可学习的仿射参数。
        权重初始化为1，偏置初始化为0。
        """
        # 初始化仿射变换参数: (num_features,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        计算并存储归一化所需的统计数据（均值和标准差，或最后一个时间步的值）。
        这些统计数据会被存储为实例属性，以便在反归一化时使用。
        :param x: torch.Tensor, 输入张量。
        """
        # 计算需要被reduce的维度，即除了batch和feature之外的所有维度（通常是序列长度维度）。
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            # 如果采用减去最后一个值的方式，则提取并保存最后一个时间步的值。
            # unsqueeze(1) 是为了保持维度一致性，便于后续广播操作。
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            # 否则，计算并保存均值。
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        # 计算并保存标准差。detach()可以防止在计算统计数据时产生梯度。
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """
        对输入张量 x 执行归一化操作。
        :param x: torch.Tensor, 输入张量。
        :return: torch.Tensor, 归一化后的张量。
        """
        if self.non_norm:
            # 如果 non_norm 为 True，则直接返回原始输入。
            return x
        if self.subtract_last:
            # 减去最后一个时间步的值。
            x = x - self.last
        else:
            # 减去均值 (Z-score 的第一步)。
            x = x - self.mean
        # 除以标准差 (Z-score 的第二步)。
        x = x / self.stdev
        if self.affine:
            # 如果启用了仿射变换，则进行缩放和偏置。
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        对输入张量 x 执行反归一化操作，恢复其原始尺度。
        :param x: torch.Tensor, 经过归一化的输入张量。
        :return: torch.Tensor, 反归一化后的张量。
        """
        if self.non_norm:
            # 如果 non_norm 为 True，则直接返回原始输入。
            return x
        if self.affine:
            # 如果启用了仿射变换，则先执行逆操作。
            x = x - self.affine_bias
            # 除以权重。加上一个小的eps防止除以零。
            x = x / (self.affine_weight + self.eps * self.eps)
        # 乘以标准差，恢复数据的尺度。
        x = x * self.stdev
        if self.subtract_last:
            # 如果之前是减去了最后一个值，现在就加回来。
            x = x + self.last
        else:
            # 否则，加上均值，恢复数据的中心。
            x = x + self.mean
        return x
