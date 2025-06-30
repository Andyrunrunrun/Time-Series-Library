# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
M4数据集处理模块

这个模块提供了M4时间序列数据集的加载、下载和处理功能。
M4数据集是一个大规模的时间序列预测基准数据集，包含来自不同领域的100,000个时间序列。

主要功能：
1. 下载M4数据集文件
2. 加载和缓存数据集
3. 提供数据集元信息
4. 支持训练集和测试集的分别加载

作者：Element AI Inc.
许可证：Creative Commons - Attribution-NonCommercial 4.0 International
"""

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import patoolib
from tqdm import tqdm
import logging
import os
import pathlib
import sys
from urllib import request


def url_file_name(url: str) -> str:
    """
    从URL中提取文件名

    这个函数用于从完整的URL路径中提取出文件名部分。
    例如：从 "https://example.com/data/file.csv" 提取出 "file.csv"

    Args:
        url (str): 需要提取文件名的URL

    Returns:
        str: 从URL中提取的文件名，如果URL为空则返回空字符串
    """
    return url.split("/")[-1] if len(url) > 0 else ""


def download(url: str, file_path: str) -> None:
    """
    下载文件到指定路径

    这个函数负责从网络下载文件并保存到本地。它包含以下功能：
    1. 显示下载进度
    2. 自动创建目录结构
    3. 避免重复下载（如果文件已存在）
    4. 记录下载日志

    Args:
        url (str): 要下载的文件URL
        file_path (str): 文件保存的本地路径
    """

    def progress(count, block_size, total_size):
        """
        下载进度回调函数

        计算并显示下载进度百分比
        """
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write(
            "\rDownloading {} to {} {:.1f}%".format(url, file_path, progress_pct)
        )
        sys.stdout.flush()

    # 检查文件是否已存在，避免重复下载
    if not os.path.isfile(file_path):
        # 设置HTTP请求头，模拟浏览器行为
        opener = request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        request.install_opener(opener)

        # 创建目标目录（如果不存在）
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)

        # 执行下载
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write("\n")
        sys.stdout.flush()

        # 记录下载成功信息
        file_info = os.stat(f)
        logging.info(
            f"Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes."
        )
    else:
        # 文件已存在，记录文件信息
        file_info = os.stat(file_path)
        logging.info(f"File already exists: {file_path} {file_info.st_size} bytes.")


@dataclass()
class M4Dataset:
    """
    M4数据集类

    这个类用于表示和加载M4时间序列数据集。它包含数据集的各个组成部分：
    - ids: 时间序列的唯一标识符
    - groups: 时间序列的季节性模式分组（年、季、月、周、日、小时）
    - frequencies: 时间序列的频率
    - horizons: 预测时域长度
    - values: 实际的时间序列数据

    Attributes:
        ids (np.ndarray): 时间序列ID数组
        groups (np.ndarray): 季节性模式分组数组
        frequencies (np.ndarray): 频率数组
        horizons (np.ndarray): 预测时域长度数组
        values (np.ndarray): 时间序列数据数组
    """

    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True, dataset_file: str = "../dataset/m4") -> "M4Dataset":
        """
        加载M4数据集

        从缓存文件中加载M4数据集。数据集被分为训练集和测试集两部分，
        通过training参数控制加载哪一部分。

        Args:
            training (bool): 如果为True则加载训练集，否则加载测试集
            dataset_file (str): 数据集文件所在的目录路径

        Returns:
            M4Dataset: 包含加载数据的M4Dataset对象
        """
        # 构建文件路径
        info_file = os.path.join(dataset_file, "M4-info.csv")  # 数据集信息文件
        train_cache_file = os.path.join(dataset_file, "training.npz")  # 训练集缓存文件
        test_cache_file = os.path.join(dataset_file, "test.npz")  # 测试集缓存文件

        # 读取数据集信息
        m4_info = pd.read_csv(info_file)

        # 根据training参数选择加载训练集或测试集
        cache_file = train_cache_file if training else test_cache_file

        # 创建并返回M4Dataset对象
        return M4Dataset(
            ids=m4_info.M4id.values,  # 时间序列ID
            groups=m4_info.SP.values,  # 季节性模式
            frequencies=m4_info.Frequency.values,  # 频率
            horizons=m4_info.Horizon.values,  # 预测时域
            values=np.load(cache_file, allow_pickle=True),  # 时间序列数据
        )


@dataclass()
class M4Meta:
    """
    M4数据集元信息类

    这个类包含了M4数据集的元信息，包括不同时间频率的配置参数。
    这些参数用于时间序列预测模型的训练和预测。

    Attributes:
        seasonal_patterns (list): 季节性模式列表
        horizons (list): 各模式的预测时域长度
        frequencies (list): 各模式的频率
        horizons_map (dict): 季节性模式到预测时域的映射
        frequency_map (dict): 季节性模式到频率的映射
        history_size (dict): 各模式的历史数据大小倍数
    """

    # 季节性模式：年、季、月、周、日、小时
    seasonal_patterns = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]

    # 各模式对应的预测时域长度
    horizons = [6, 8, 18, 13, 14, 48]

    # 各模式对应的频率（每年/季/月/周/日/小时的数据点数量）
    frequencies = [1, 4, 12, 1, 1, 24]

    # 季节性模式到预测时域的映射字典
    horizons_map = {
        "Yearly": 6,  # 年度数据预测6个点
        "Quarterly": 8,  # 季度数据预测8个点
        "Monthly": 18,  # 月度数据预测18个点
        "Weekly": 13,  # 周度数据预测13个点
        "Daily": 14,  # 日度数据预测14个点
        "Hourly": 48,  # 小时数据预测48个点
    }

    # 季节性模式到频率的映射字典
    frequency_map = {
        "Yearly": 1,  # 年度数据频率为1（每年1个数据点）
        "Quarterly": 4,  # 季度数据频率为4（每年4个数据点）
        "Monthly": 12,  # 月度数据频率为12（每年12个数据点）
        "Weekly": 1,  # 周度数据频率为1（每周1个数据点）
        "Daily": 1,  # 日度数据频率为1（每天1个数据点）
        "Hourly": 24,  # 小时数据频率为24（每天24个数据点）
    }

    # 各模式的历史数据大小倍数（用于确定训练时使用多少历史数据）
    history_size = {
        "Yearly": 1.5,  # 年度数据使用1.5倍预测时域的历史数据
        "Quarterly": 1.5,  # 季度数据使用1.5倍预测时域的历史数据
        "Monthly": 1.5,  # 月度数据使用1.5倍预测时域的历史数据
        "Weekly": 10,  # 周度数据使用10倍预测时域的历史数据
        "Daily": 10,  # 日度数据使用10倍预测时域的历史数据
        "Hourly": 10,  # 小时数据使用10倍预测时域的历史数据
    }


def load_m4_info() -> pd.DataFrame:
    """
    加载M4数据集信息文件

    这个函数用于加载M4数据集的元信息文件，该文件包含了所有时间序列的
    基本信息，如ID、季节性模式、频率、预测时域等。

    Returns:
        pd.DataFrame: 包含M4数据集信息的Pandas DataFrame

    Note:
        此函数目前存在未定义变量INFO_FILE_PATH的问题，需要修复
    """
    print("==========INFO_FILE_PATH==========", info_file_path)
    return pd.read_csv(info_file_path)
