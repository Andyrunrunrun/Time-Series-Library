# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
时间特征提取工具模块

这个模块提供了从时间序列数据中提取时间特征的功能，主要用于时间序列预测和建模。
时间特征包括：秒、分钟、小时、星期几、月份、年份等周期性特征。

主要功能：
1. 将时间信息编码为数值特征（范围通常在[-0.5, 0.5]之间）
2. 根据数据频率自动选择合适的特征
3. 支持多种时间频率：年、月、周、日、小时、分钟、秒
"""

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """
    时间特征基类

    所有具体的时间特征类都继承自这个基类。
    定义了时间特征提取的标准接口。
    """

    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        将时间特征类作为可调用对象使用

        Args:
            index: pandas DatetimeIndex对象，包含时间戳信息

        Returns:
            numpy数组，包含提取的时间特征值
        """
        return np.array([])  # 基类返回空数组，子类会重写此方法

    def __repr__(self):
        """返回类的字符串表示"""
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """
    分钟内的秒数特征

    将分钟内的秒数编码为[-0.5, 0.5]范围内的值。
    例如：0秒 -> -0.5, 30秒 -> 0, 59秒 -> 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        提取分钟内的秒数特征

        Args:
            index: 时间索引

        Returns:
            秒数特征数组，值域为[-0.5, 0.5]
        """
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """
    小时内的分钟数特征

    将小时内的分钟数编码为[-0.5, 0.5]范围内的值。
    例如：0分钟 -> -0.5, 30分钟 -> 0, 59分钟 -> 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        提取小时内的分钟数特征

        Args:
            index: 时间索引

        Returns:
            分钟数特征数组，值域为[-0.5, 0.5]
        """
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """
    一天内的小时数特征

    将一天内的小时数编码为[-0.5, 0.5]范围内的值。
    例如：0点 -> -0.5, 12点 -> 0, 23点 -> 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        提取一天内的小时数特征

        Args:
            index: 时间索引

        Returns:
            小时数特征数组，值域为[-0.5, 0.5]
        """
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """
    一周内的星期几特征

    将星期几编码为[-0.5, 0.5]范围内的值。
    例如：周一 -> -0.5, 周四 -> 0, 周日 -> 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        提取星期几特征

        Args:
            index: 时间索引

        Returns:
            星期几特征数组，值域为[-0.5, 0.5]
        """
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """
    一个月内的天数特征

    将一个月内的天数编码为[-0.5, 0.5]范围内的值。
    例如：1号 -> -0.5, 15号 -> 0, 31号 -> 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        提取一个月内的天数特征

        Args:
            index: 时间索引

        Returns:
            天数特征数组，值域为[-0.5, 0.5]
        """
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """
    一年内的天数特征

    将一年内的天数编码为[-0.5, 0.5]范围内的值。
    例如：1月1日 -> -0.5, 7月2日 -> 0, 12月31日 -> 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        提取一年内的天数特征

        Args:
            index: 时间索引

        Returns:
            年天数特征数组，值域为[-0.5, 0.5]
        """
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """
    一年内的月份特征

    将一年内的月份编码为[-0.5, 0.5]范围内的值。
    例如：1月 -> -0.5, 7月 -> 0, 12月 -> 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        提取一年内的月份特征

        Args:
            index: 时间索引

        Returns:
            月份特征数组，值域为[-0.5, 0.5]
        """
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """
    一年内的周数特征

    将一年内的周数编码为[-0.5, 0.5]范围内的值。
    例如：第1周 -> -0.5, 第27周 -> 0, 第52周 -> 0.5
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        提取一年内的周数特征

        Args:
            index: 时间索引

        Returns:
            周数特征数组，值域为[-0.5, 0.5]
        """
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    根据频率字符串返回合适的时间特征列表

    这个函数根据数据的时间频率自动选择合适的特征提取器。
    例如：小时级数据会提取小时、星期几、月份等特征。

    Parameters
    ----------
    freq_str : str
        频率字符串，格式为[倍数][粒度]，如"12H"、"5min"、"1D"等

    Returns
    -------
    List[TimeFeature]
        适合该频率的时间特征提取器列表

    Raises
    ------
    RuntimeError
        当频率不被支持时抛出异常
    """

    # 定义不同时间频率对应的特征提取器
    features_by_offsets = {
        offsets.YearEnd: [],  # 年度数据：不需要额外特征
        offsets.QuarterEnd: [MonthOfYear],  # 季度数据：月份特征
        offsets.MonthEnd: [MonthOfYear],  # 月度数据：月份特征
        offsets.Week: [DayOfMonth, WeekOfYear],  # 周度数据：月内天数、年周数
        offsets.Day: [
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],  # 日度数据：星期几、月内天数、年天数
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],  # 工作日数据：同上
        offsets.Hour: [
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],  # 小时数据：小时、星期几、月内天数、年天数
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],  # 分钟数据：分钟、小时、星期几、月内天数、年天数
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],  # 秒级数据：所有特征
    }

    # 将频率字符串转换为pandas偏移对象
    offset = to_offset(freq_str)

    # 根据偏移类型返回对应的特征提取器
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    # 如果不支持的频率，抛出异常并显示支持的类型
    supported_freq_msg = f"""
    不支持的时间频率: {freq_str}
    支持的时间频率包括:
        Y   - 年度
            别名: A
        M   - 月度
        W   - 周度
        D   - 日度
        B   - 工作日
        H   - 小时
        T   - 分钟
            别名: min
        S   - 秒级
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq="h"):
    """
    从时间索引中提取时间特征

    这是一个便捷函数，将时间索引转换为数值特征矩阵。

    Parameters
    ----------
    dates : pd.DatetimeIndex
        时间索引
    freq : str, default='h'
        时间频率字符串

    Returns
    -------
    np.ndarray
        时间特征矩阵，每行对应一个时间点，每列对应一个特征
    """
    # 获取适合该频率的特征提取器列表
    feature_extractors = time_features_from_frequency_str(freq)

    # 对每个特征提取器调用，并将结果垂直堆叠成矩阵
    return np.vstack([feat(dates) for feat in feature_extractors])
