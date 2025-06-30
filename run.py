import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

if __name__ == "__main__":
    # 固定随机种子，保证实验可复现
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="TimesNet")

    # 基本配置
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        default="long_term_forecast",
        help="任务名称，选项:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="训练状态，1为训练，0为测试"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="模型ID"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="模型名称，选项: [Autoformer, Transformer, TimesNet]",
    )

    # 数据加载相关参数
    parser.add_argument(
        "--data", type=str, required=True, default="ETTh1", help="数据集类型"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="数据文件根目录",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="数据文件名")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="预测任务类型，选项:[M, S, MS]; M:多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="S或MS任务中的目标特征"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="时间特征编码的频率，选项:[s:秒, t:分钟, h:小时, d:天, b:工作日, w:周, m:月]，也可用更细致的如15min或3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="模型检查点保存路径",
    )

    # 预测任务相关参数
    parser.add_argument("--seq_len", type=int, default=96, help="输入序列长度")
    parser.add_argument("--label_len", type=int, default=48, help="标签序列长度")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="预测序列长度"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="M4数据集的子集"
    )
    parser.add_argument(
        "--inverse", action="store_true", help="是否对输出数据做逆变换", default=False
    )

    # 插值任务相关参数
    parser.add_argument("--mask_rate", type=float, default=0.25, help="掩码比例")

    # 异常检测任务相关参数
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="异常比例(%)"
    )

    # 模型结构相关参数
    parser.add_argument(
        "--expand", type=int, default=2, help="Mamba模型的扩展因子"
    )
    parser.add_argument(
        "--d_conv", type=int, default=4, help="Mamba模型的卷积核大小"
    )
    parser.add_argument("--top_k", type=int, default=5, help="TimesBlock的top_k")
    parser.add_argument("--num_kernels", type=int, default=6, help="Inception的核数")
    parser.add_argument("--enc_in", type=int, default=7, help="编码器输入维度")
    parser.add_argument("--dec_in", type=int, default=7, help="解码器输入维度")
    parser.add_argument("--c_out", type=int, default=7, help="输出维度")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--n_heads", type=int, default=8, help="多头注意力头数")
    parser.add_argument("--e_layers", type=int, default=2, help="编码器层数")
    parser.add_argument("--d_layers", type=int, default=1, help="解码器层数")
    parser.add_argument("--d_ff", type=int, default=2048, help="全连接层维度")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="滑动平均窗口大小"
    )
    parser.add_argument("--factor", type=int, default=1, help="注意力因子")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="编码器是否使用蒸馏，使用该参数表示不使用蒸馏",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout比例")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="时间特征编码方式，选项:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="激活函数")
    parser.add_argument(
        "--channel_independence",
        type=int,
        default=1,
        help="FreTS模型的通道独立性，0:相关，1:独立",
    )
    parser.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        help="序列分解方法，仅支持moving_avg或dft_decomp",
    )
    parser.add_argument(
        "--use_norm",
        type=int,
        default=1,
        help="是否归一化，1为是，0为否",
    )
    parser.add_argument(
        "--down_sampling_layers",
        type=int,
        default=0,
        help="下采样层数",
    )
    parser.add_argument(
        "--down_sampling_window", type=int, default=1, help="下采样窗口大小"
    )
    parser.add_argument(
        "--down_sampling_method",
        type=str,
        default=None,
        help="下采样方法，仅支持avg, max, conv",
    )
    parser.add_argument(
        "--seg_len",
        type=int,
        default=96,
        help="SegRNN的分段长度",
    )

    # 优化器相关参数
    parser.add_argument(
        "--num_workers", type=int, default=10, help="数据加载线程数"
    )
    parser.add_argument("--itr", type=int, default=1, help="实验重复次数")
    parser.add_argument("--train_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="训练批次大小"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="早停耐心值"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="学习率"
    )
    parser.add_argument("--des", type=str, default="test", help="实验描述")
    parser.add_argument("--loss", type=str, default="MSE", help="损失函数")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="学习率调整方式"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="是否使用自动混合精度训练",
        default=False,
    )

    # GPU相关参数
    parser.add_argument("--use_gpu", type=bool, default=True, help="是否使用GPU")
    parser.add_argument("--gpu", type=int, default=0, help="GPU编号")
    parser.add_argument(
        "--gpu_type", type=str, default="cuda", help="GPU类型，cuda或mps"
    )
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="是否使用多GPU", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="多GPU设备编号"
    )

    # 非平稳投影器参数
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="投影器隐藏层维度（列表）",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="投影器隐藏层数",
    )

    # DTW指标
    parser.add_argument(
        "--use_dtw",
        type=bool,
        default=False,
        help="是否使用dtw指标（dtw计算耗时，非必要不建议开启）",
    )

    # 数据增强相关参数
    parser.add_argument(
        "--augmentation_ratio", type=int, default=0, help="增强倍数"
    )
    parser.add_argument("--seed", type=int, default=2, help="随机种子")
    parser.add_argument(
        "--jitter",
        default=False,
        action="store_true",
        help="Jitter预设增强",
    )
    parser.add_argument(
        "--scaling",
        default=False,
        action="store_true",
        help="Scaling预设增强",
    )
    parser.add_argument(
        "--permutation",
        default=False,
        action="store_true",
        help="等长置换预设增强",
    )
    parser.add_argument(
        "--randompermutation",
        default=False,
        action="store_true",
        help="随机长度置换预设增强",
    )
    parser.add_argument(
        "--magwarp",
        default=False,
        action="store_true",
        help="幅度扰动预设增强",
    )
    parser.add_argument(
        "--timewarp",
        default=False,
        action="store_true",
        help="时间扰动预设增强",
    )
    parser.add_argument(
        "--windowslice",
        default=False,
        action="store_true",
        help="窗口切片预设增强",
    )
    parser.add_argument(
        "--windowwarp",
        default=False,
        action="store_true",
        help="窗口扰动预设增强",
    )
    parser.add_argument(
        "--rotation",
        default=False,
        action="store_true",
        help="旋转预设增强",
    )
    parser.add_argument(
        "--spawner",
        default=False,
        action="store_true",
        help="SPAWNER预设增强",
    )
    parser.add_argument(
        "--dtwwarp",
        default=False,
        action="store_true",
        help="DTW扰动预设增强",
    )
    parser.add_argument(
        "--shapedtwwarp",
        default=False,
        action="store_true",
        help="Shape DTW扰动预设增强",
    )
    parser.add_argument(
        "--wdba",
        default=False,
        action="store_true",
        help="加权DBA预设增强",
    )
    parser.add_argument(
        "--discdtw",
        default=False,
        action="store_true",
        help="判别性DTW扰动预设增强",
    )
    parser.add_argument(
        "--discsdtw",
        default=False,
        action="store_true",
        help="判别性ShapeDTW扰动预设增强",
    )
    parser.add_argument("--extra_tag", type=str, default="", help="额外标签")

    # TimeXer相关参数
    parser.add_argument("--patch_len", type=int, default=16, help="patch长度")

    # 解析命令行参数
    args = parser.parse_args()

    # 设备选择：优先使用GPU，否则使用MPS或CPU
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device("cuda:{}".format(args.gpu))
        print("Using GPU")
    else:
        if hasattr(torch.backends, "mps"):
            args.device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            args.device = torch.device("cpu")
        print("Using cpu or mps")

    # 多GPU设置
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 打印实验参数
    print("Args in experiment:")
    print_args(args)

    # 根据任务名称选择实验类
    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == "imputation":
        Exp = Exp_Imputation
    elif args.task_name == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args.task_name == "classification":
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    # 训练或测试流程
    if args.is_training:
        for ii in range(args.itr):
            # 设置实验
            exp = Exp(args)  # 实例化实验对象
            # 构建实验设置字符串，便于记录和区分不同实验
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii,
            )

            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)  # 开始训练

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)  # 训练后测试
            # 清理显存
            if args.gpu_type == "mps":
                torch.backends.mps.empty_cache()
            elif args.gpu_type == "cuda":
                torch.cuda.empty_cache()
    else:
        # 只做测试
        exp = Exp(args)  # 实例化实验对象
        ii = 0
        setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)  # 只测试
        # 清理显存
        if args.gpu_type == "mps":
            torch.backends.mps.empty_cache()
        elif args.gpu_type == "cuda":
            torch.cuda.empty_cache()