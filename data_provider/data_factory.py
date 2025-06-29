"""
数据工厂模块 - 用于创建和管理各种时间序列数据集的加载器

该模块提供了统一的数据加载接口，支持多种时间序列数据集：
- ETT数据集：电力变压器数据（小时级和分钟级）
- 自定义数据集：用户自定义的时间序列数据
- M4数据集：M4时间序列预测竞赛数据集
- 异常检测数据集：PSM、MSL、SMAP、SMD、SWAT等
- UEA数据集：UEA时间序列分类数据集
"""

from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_M4,
    PSMSegLoader,
    MSLSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    SWATSegLoader,
    UEAloader,
)
from data_provider.uea import collate_fn
from torch.utils.data.dataloader import DataLoader

# 数据集字典：将数据集名称映射到对应的数据集类
data_dict = {
    "ETTh1": Dataset_ETT_hour,  # ETT数据集 - 小时级数据，包含1个变压器的数据
    "ETTh2": Dataset_ETT_hour,  # ETT数据集 - 小时级数据，包含2个变压器的数据
    "ETTm1": Dataset_ETT_minute,  # ETT数据集 - 分钟级数据，包含1个变压器的数据
    "ETTm2": Dataset_ETT_minute,  # ETT数据集 - 分钟级数据，包含2个变压器的数据
    "custom": Dataset_Custom,  # 自定义数据集 - 用户提供的时间序列数据
    "m4": Dataset_M4,  # M4数据集 - 时间序列预测竞赛数据集，包含100,000个时间序列
    "PSM": PSMSegLoader,  # PSM数据集 - 服务器性能监控数据，用于异常检测
    "MSL": MSLSegLoader,  # MSL数据集 - 火星科学实验室数据，用于异常检测
    "SMAP": SMAPSegLoader,  # SMAP数据集 - 土壤水分主动被动卫星数据，用于异常检测
    "SMD": SMDSegLoader,  # SMD数据集 - 服务器机器数据集，用于异常检测
    "SWAT": SWATSegLoader,  # SWAT数据集 - 安全水处理系统数据，用于异常检测
    "UEA": UEAloader,  # UEA数据集 - 时间序列分类数据集集合
}


def data_provider(args, flag):
    """
    数据提供者函数 - 根据任务类型和数据集创建相应的数据加载器

    参数:
        args: 配置参数对象，包含数据集名称、任务类型、序列长度等配置
        flag: 数据分割标志，可以是 'train'、'val'、'test' 等

    返回:
        data_set: 数据集对象
        data_loader: PyTorch数据加载器对象
    """
    # 根据数据集名称获取对应的数据集类
    Data = data_dict[args.data]

    # 时间编码设置：如果嵌入类型是'timeF'则使用时间特征编码，否则不使用
    timeenc = 0 if args.embed != "timeF" else 1

    # 设置数据加载参数
    # 测试集不打乱顺序，训练集和验证集打乱顺序
    shuffle_flag = False if (flag == "test" or flag == "TEST") else True
    drop_last = False  # 默认不丢弃最后一个不完整的批次
    batch_size = args.batch_size
    freq = args.freq

    # 异常检测任务的数据加载器创建
    if args.task_name == "anomaly_detection":
        drop_last = False
        # 创建异常检测数据集
        data_set = Data(
            args=args,
            root_path=args.root_path,  # 数据根路径
            win_size=args.seq_len,  # 滑动窗口大小（序列长度）
            flag=flag,  # 数据分割标志
        )
        print(flag, len(data_set))  # 打印数据集分割类型和大小

        # 创建PyTorch数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,  # 批次大小
            shuffle=shuffle_flag,  # 是否打乱数据
            num_workers=args.num_workers,  # 数据加载的工作进程数
            drop_last=drop_last,
        )  # 是否丢弃最后一个不完整的批次
        return data_set, data_loader

    # 分类任务的数据加载器创建
    elif args.task_name == "classification":
        drop_last = False
        # 创建分类数据集
        data_set = Data(
            args=args,
            root_path=args.root_path,  # 数据根路径
            flag=flag,  # 数据分割标志
        )

        # 创建PyTorch数据加载器，使用自定义的collate函数处理变长序列
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,  # 批次大小
            shuffle=shuffle_flag,  # 是否打乱数据
            num_workers=args.num_workers,  # 数据加载的工作进程数
            drop_last=drop_last,  # 是否丢弃最后一个不完整的批次
            collate_fn=lambda x: collate_fn(
                x, max_len=args.seq_len
            ),  # 自定义数据整理函数
        )
        return data_set, data_loader

    # 时间序列预测任务的数据加载器创建（默认情况）
    else:
        # M4数据集特殊处理：不丢弃最后一个批次
        if args.data == "m4":
            drop_last = False

        # 创建时间序列预测数据集
        data_set = Data(
            args=args,
            root_path=args.root_path,  # 数据根路径
            data_path=args.data_path,  # 数据文件路径
            flag=flag,  # 数据分割标志
            size=[
                args.seq_len,
                args.label_len,
                args.pred_len,
            ],  # [输入序列长度, 标签长度, 预测长度]
            features=args.features,  # 特征类型（'M'多变量, 'S'单变量, 'MS'多变量单目标）
            target=args.target,  # 目标变量名称
            timeenc=timeenc,  # 时间编码类型
            freq=freq,  # 数据频率
            seasonal_patterns=args.seasonal_patterns,  # 季节性模式（M4数据集专用）
        )
        print(flag, len(data_set))  # 打印数据集分割类型和大小

        # 创建PyTorch数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,  # 批次大小
            shuffle=shuffle_flag,  # 是否打乱数据
            num_workers=args.num_workers,  # 数据加载的工作进程数
            drop_last=drop_last,
        )  # 是否丢弃最后一个不完整的批次
        return data_set, data_loader


"""
数据集详细信息说明
==================

1. ETT数据集 (Electricity Transformer Temperature)
   - 描述：电力变压器温度数据，包含多个变压器的运行数据
   - 数据特点：
     * ETTh1/ETTh2：小时级数据，分别包含1个和2个变压器的数据
     * ETTm1/ETTm2：分钟级数据，分别包含1个和2个变压器的数据
   - 变量：包含负载、温度、湿度等7个变量
   - 应用场景：电力系统预测、设备故障预测
   - 数据量：约17,420个时间点（小时级），约696,800个时间点（分钟级）

2. M4数据集
   - 描述：M4时间序列预测竞赛数据集，包含100,000个时间序列
   - 数据特点：
     * 包含年度、季度、月度、周度、日度、小时度等不同频率的数据
     * 涵盖经济、人口、工业、金融等多个领域
   - 应用场景：时间序列预测研究、模型性能评估
   - 数据量：100,000个时间序列，预测长度从6到48不等

3. 异常检测数据集
   
   a) PSM数据集 (Pooled Server Metrics)
      - 描述：服务器性能监控数据
      - 数据特点：包含CPU使用率、内存使用率、网络流量等指标
      - 应用场景：服务器异常检测、IT运维监控
      - 数据量：约132,481个时间点，38个变量
   
   b) MSL数据集 (Mars Science Laboratory)
      - 描述：火星科学实验室探测器数据
      - 数据特点：包含传感器读数、设备状态等航天器运行数据
      - 应用场景：航天器异常检测、设备健康监控
      - 数据量：约58,317个时间点，55个变量
   
   c) SMAP数据集 (Soil Moisture Active Passive)
      - 描述：土壤水分主动被动卫星数据
      - 数据特点：包含土壤湿度、温度等地球观测数据
      - 应用场景：环境监测、农业预测
      - 数据量：约135,183个时间点，25个变量
   
   d) SMD数据集 (Server Machine Dataset)
      - 描述：服务器机器数据集
      - 数据特点：包含服务器硬件和软件性能指标
      - 应用场景：服务器故障预测、IT基础设施监控
      - 数据量：约708,405个时间点，38个变量
   
   e) SWAT数据集 (Secure Water Treatment)
      - 描述：安全水处理系统数据
      - 数据特点：包含水处理过程中的各种传感器数据
      - 应用场景：工业过程监控、水处理系统异常检测
      - 数据量：约946,722个时间点，51个变量

4. UEA数据集 (University of East Anglia)
   - 描述：时间序列分类数据集集合
   - 数据特点：
     * 包含多种类型的时间序列分类任务
     * 涵盖语音识别、手势识别、心电图分析等领域
   - 应用场景：时间序列分类、模式识别
   - 数据量：包含128个不同的数据集

5. 自定义数据集 (Custom)
   - 描述：用户自定义的时间序列数据
   - 数据特点：支持用户提供自己的时间序列数据
   - 应用场景：特定领域的时间序列分析
   - 数据量：根据用户提供的数据而定

任务类型说明
============

1. 异常检测 (anomaly_detection)
   - 目标：检测时间序列中的异常模式
   - 适用数据集：PSM、MSL、SMAP、SMD、SWAT
   - 特点：使用滑动窗口方法，需要指定窗口大小

2. 分类 (classification)
   - 目标：对时间序列进行分类
   - 适用数据集：UEA数据集
   - 特点：使用自定义的collate函数处理变长序列

3. 时间序列预测 (forecasting)
   - 目标：预测时间序列的未来值
   - 适用数据集：ETT、M4、自定义数据集
   - 特点：需要指定输入序列长度、标签长度和预测长度

参数说明
========

- seq_len: 输入序列长度
- label_len: 标签序列长度（用于训练）
- pred_len: 预测序列长度
- features: 特征类型
  * 'M': 多变量预测
  * 'S': 单变量预测
  * 'MS': 多变量单目标预测
- target: 目标变量名称
- freq: 数据频率（如'h'表示小时，'m'表示分钟）
- embed: 时间编码类型
  * 'timeF': 使用时间特征编码
  * 其他: 不使用时间编码
"""
