"""
UEA时间序列数据处理工具库

这个模块提供了处理UEA（University of East Anglia）时间序列数据集的工具函数，
包括数据批处理、标准化、缺失值处理等功能。主要用于深度学习模型的数据预处理。
"""

import os
import numpy as np
import pandas as pd
import torch


def collate_fn(data, max_len=None):
    """
    将时间序列数据列表转换为批次张量，处理变长序列的填充问题

    这个函数是PyTorch DataLoader的collate_fn参数，用于将不同长度的
    时间序列数据打包成固定长度的批次，以便输入到深度学习模型中。

    Args:
        data: 长度为batch_size的元组列表，每个元组包含(X, y)
            - X: torch张量，形状为(seq_length, feat_dim)，序列长度可变
            - y: torch张量，形状为(num_labels,)，类别索引或数值目标
                (用于分类或回归)，num_labels > 1用于多任务模型
        max_len: 全局固定序列长度。用于需要固定长度输入的架构，
                其中批次长度不能动态变化。较长序列会被截断，较短序列用0填充

    Returns:
        X: (batch_size, padded_length, feat_dim) torch张量，填充后的特征张量（输入）
        targets: (batch_size, num_labels) torch张量，标签张量
        padding_masks: (batch_size, padded_length) 布尔张量，1表示保留该位置的向量，0表示填充
    """

    batch_size = len(data)
    features, labels = zip(*data)  # 解包特征和标签

    # 堆叠和填充特征和掩码（将2D张量转换为3D张量，即添加批次维度）
    lengths = [X.shape[0] for X in features]  # 每个时间序列的原始序列长度
    if max_len is None:
        max_len = max(lengths)  # 如果没有指定最大长度，使用最长序列的长度

    # 创建填充后的特征张量
    X = torch.zeros(
        batch_size, max_len, features[0].shape[-1]
    )  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)  # 确定当前序列的结束位置
        X[i, :end, :] = features[i][:end, :]  # 填充真实数据，其余位置保持为0

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels) 堆叠标签

    # 创建填充掩码，标识哪些位置是真实数据
    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16), max_len=max_len
    )  # (batch_size, padded_length) 布尔张量，"1"表示保留

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    创建填充位置掩码：从序列长度张量创建(batch_size, max_len)布尔掩码

    这个函数用于标识哪些位置是真实数据，哪些是填充数据。
    在训练时，模型应该忽略填充部分，只关注真实数据。

    Args:
        lengths: 包含每个序列长度的张量
        max_len: 最大序列长度，如果为None则使用lengths中的最大值

    Returns:
        mask: (batch_size, max_len) 布尔张量，1表示保留该位置的元素（时间步）
    """
    batch_size = lengths.numel()  # 批次大小
    max_len = max_len or lengths.max_val()  # 技巧：利用'or'运算符对非布尔类型的重载
    return (
        torch.arange(0, max_len, device=lengths.device)  # 创建位置索引
        .type_as(lengths)  # 保持与lengths相同的数据类型
        .repeat(batch_size, 1)  # 重复到批次大小
        .lt(lengths.unsqueeze(1))  # 比较：位置 < 序列长度，生成布尔掩码
    )


class Normalizer(object):
    """
    数据标准化器：对所有包含的行（时间步）进行标准化

    与按样本标准化不同，这个类对整个数据框进行全局标准化。
    支持多种标准化方法，包括标准化、最小-最大归一化等。
    """

    def __init__(
        self,
        norm_type="standardization",
        mean=None,
        std=None,
        min_val=None,
        max_val=None,
    ):
        """
        初始化标准化器

        Args:
            norm_type: 标准化类型，可选：
                "standardization": 标准化（Z-score标准化）
                "minmax": 最小-最大归一化
                "per_sample_std": 按样本标准化（每个样本单独标准化）
                "per_sample_minmax": 按样本最小-最大归一化
            mean, std, min_val, max_val: 可选的预计算统计值，形状为(num_feat,)的Series
        """

        self.norm_type = norm_type  # 标准化类型
        self.mean = mean  # 均值
        self.std = std  # 标准差
        self.min_val = min_val  # 最小值
        self.max_val = max_val  # 最大值

    def normalize(self, df):
        """
        对数据框进行标准化

        Args:
            df: 输入数据框

        Returns:
            df: 标准化后的数据框
        """
        if self.norm_type == "standardization":
            # 标准化：(x - mean) / std
            if self.mean is None:
                self.mean = df.mean()  # 计算均值
                self.std = df.std()  # 计算标准差
            if self.std is None:
                raise ValueError("Standard deviation is not set")  # 标准差未设置
            # 添加eps避免除零错误
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            # 最小-最大归一化：(x - min) / (max - min)
            if self.max_val is None:
                self.max_val = df.max()  # 计算最大值
                self.min_val = df.min()  # 计算最小值
            # 添加eps避免除零错误
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            # 按样本标准化：每个样本单独进行标准化
            grouped = df.groupby(by=df.index)  # 按索引分组
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            # 按样本最小-最大归一化：每个样本单独进行归一化
            grouped = df.groupby(by=df.index)  # 按索引分组
            min_vals = grouped.transform("min")  # 每个样本的最小值
            # 添加eps避免除零错误
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            # 不支持的标准化方法
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    使用线性插值填充pd.Series中的NaN值

    这个函数用于处理时间序列中的缺失值，通过线性插值的方法
    在缺失值的前后数据点之间进行插值填充。

    Args:
        y: 包含可能缺失值的时间序列

    Returns:
        y: 插值填充后的时间序列
    """
    if y.isna().any():  # 检查是否有缺失值
        # 使用线性插值，双向填充（向前和向后）
        y = y.interpolate(method="linear", limit_direction="both")
    return y


def subsample(y, limit=256, factor=2):
    """
    对时间序列进行降采样

    如果给定的Series长度超过limit，则按指定的整数因子进行降采样。
    这通常用于处理过长的时间序列，减少计算复杂度。

    Args:
        y: 输入时间序列
        limit: 长度限制，默认256
        factor: 降采样因子，默认2（即每隔factor个点取一个点）

    Returns:
        y: 降采样后的时间序列，如果原序列长度不超过limit则返回原序列
    """
    if len(y) > limit:  # 如果序列长度超过限制
        return y[::factor].reset_index(drop=True)  # 按因子降采样并重置索引
    return y  # 否则返回原序列
