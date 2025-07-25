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
import pandas as pd
from datetime import datetime

def save_results_to_csv(args, results_list, final_stats, filename=None):
    """
    将实验参数和结果保存到CSV文件中，支持追加模式。
    
    该函数实现了完整的实验结果记录功能，包括：
    - 实验超参数的完整记录
    - 每次迭代的详细指标结果
    - 支持多次实验的追加保存
    - 自动计算统计汇总信息
    - 根据任务类型保存到不同的CSV文件
    
    Args:
        args (Namespace): 实验参数对象，包含所有超参数配置
        results_list (list): 包含每次迭代结果的列表，每个元素为字典格式
        final_stats (dict): 最终统计结果，包含均值和标准差
        filename (str): CSV文件名，如果为None则根据任务类型自动生成
        
    Returns:
        str: 保存的CSV文件路径
        
    Example:
        >>> args = get_parsed_args()
        >>> results = [{'mse': 0.1, 'mae': 0.08, 'train_time': 120.5}, ...]
        >>> stats = {'mse_mean': 0.12, 'mse_std': 0.02, 'mae_mean': 0.09, 'mae_std': 0.01}
        >>> filename = save_results_to_csv(args, results, stats)
        >>> print(f"结果已保存到: {filename}")
    """
    # (1) 创建log文件夹
    log_folder = "log"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
        print(f"创建log文件夹: {log_folder}")
    
    # (2) 根据任务类型生成文件名
    if filename is None:
        task_file_mapping = {
            'long_term_forecast': 'long_term_forecasting_results.csv',
            'short_term_forecast': 'short_term_forecasting_results.csv',
            'anomaly_detection': 'anomaly_detection_results.csv',
            'classification': 'classification_results.csv',
            'imputation': 'imputation_results.csv'
        }
        filename = task_file_mapping.get(args.task_name, 'unknown_task_results.csv')
    
    # (3) 完整的文件路径
    filepath = os.path.join(log_folder, filename)
    
    # (4) 准备实验参数数据 - 提取所有重要的超参数
    params_dict = {
        'task_name': args.task_name,
        'model_id': args.model_id,
        'model': args.model,
        'data': args.data,
        'data_path': args.data_path,
        'features': args.features,
        'target': args.target,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'label_len': args.label_len,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'train_epochs': args.train_epochs,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'e_layers': args.e_layers,
        'd_layers': args.d_layers,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'embed': args.embed,
        'freq': args.freq,
        'enc_in': args.enc_in,
        'c_out': args.c_out,
        'loss': args.loss,
        'lradj': args.lradj,
        'patience': args.patience,
        'use_amp': args.use_amp,
        'use_gpu': args.use_gpu,
        'gpu': args.gpu,
        'itr': args.itr
    }
    
    # (5) 创建当前实验的详细结果数据 - 包含每次迭代的完整信息
    current_experiment_data = []
    for i, result in enumerate(results_list):
        row = params_dict.copy()  # 复制参数字典
        row.update({
            'iteration': i + 1,
            'train_time': result.get('train_time', 0),
            'final_train_loss': result.get('final_train_loss', 0),
            'final_vali_loss': result.get('final_vali_loss', 0),
            'epochs_trained': result.get('epochs_trained', 0),
            'experiment_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 根据任务类型添加相应的指标
        if args.task_name == 'long_term_forecast':
            # 长期预测任务使用mse, mae等指标
            row.update({
                'mse': result.get('mse', 0),
                'mae': result.get('mae', 0),
                'rmse': result.get('rmse', 0),
                'mape': result.get('mape', 0),
                'mspe': result.get('mspe', 0),
                'dtw': result.get('dtw', 'Not calculated')
            })
        elif args.task_name == 'imputation':
            # 插值任务使用mse, mae等指标，额外包含mask_rate
            row.update({
                'mse': result.get('mse', 0),
                'mae': result.get('mae', 0),
                'rmse': result.get('rmse', 0),
                'mape': result.get('mape', 0),
                'mspe': result.get('mspe', 0),
                'mask_rate': result.get('mask_rate', args.mask_rate)
            })
        elif args.task_name == 'short_term_forecast':
            # 短期预测任务使用M4指标
            row.update({
                'smape': result.get('smape', 0),
                'owa': result.get('owa', 0),
                'mape': result.get('mape', 0),
                'mase': result.get('mase', 0),
                'seasonal_patterns': result.get('seasonal_patterns', ''),
                'status': result.get('status', 'completed')
            })
        elif args.task_name == 'anomaly_detection':
            # 异常检测任务使用分类指标
            row.update({
                'accuracy': result.get('accuracy', 0),
                'precision': result.get('precision', 0),
                'recall': result.get('recall', 0),
                'f_score': result.get('f_score', 0),
                'threshold': result.get('threshold', 0),
                'anomaly_ratio': result.get('anomaly_ratio', args.anomaly_ratio)
            })
        elif args.task_name == 'classification':
            # 分类任务使用准确率指标
            row.update({
                'accuracy': result.get('accuracy', 0),
                'num_classes': result.get('num_classes', 0),
                'total_samples': result.get('total_samples', 0)
            })
        
        current_experiment_data.append(row)
    
    # (6) 处理文件追加逻辑
    if os.path.exists(filepath):
        # 如果文件已存在，读取现有数据并追加新实验
        print(f"检测到现有文件 {filepath}，将追加新的{args.task_name}实验结果...")
        try:
            existing_df = pd.read_csv(filepath, encoding='utf-8-sig')
            # 删除之前的SUMMARY行（如果存在）
            existing_df = existing_df[existing_df['iteration'] != 'SUMMARY'].copy()
            
            # 将新实验数据追加到现有数据
            all_data = existing_df.to_dict('records') + current_experiment_data
            
            print(f"成功追加 {len(current_experiment_data)} 条新记录到现有的 {len(existing_df)} 条记录中")
        except Exception as e:
            print(f"读取现有文件时出错: {e}，将创建新文件")
            all_data = current_experiment_data
    else:
        # 如果文件不存在，创建新文件
        print(f"创建新的{args.task_name}实验结果文件: {filepath}")
        all_data = current_experiment_data
    
    # (7) 计算整体统计信息 - 基于所有历史数据
    primary_metrics = []  # 存储主要指标
    secondary_metrics = []  # 存储次要指标
    experiment_groups = {}  # 按实验时间戳分组
    
    # 根据任务类型确定主要指标
    for record in all_data:
        if isinstance(record['iteration'], int):  # 排除SUMMARY行
            if args.task_name == 'long_term_forecast' or args.task_name == 'imputation':
                primary_metrics.append(record.get('mse', 0))
                secondary_metrics.append(record.get('mae', 0))
            elif args.task_name == 'short_term_forecast':
                primary_metrics.append(record.get('smape', 0))
                secondary_metrics.append(record.get('owa', 0))
            elif args.task_name == 'anomaly_detection':
                primary_metrics.append(record.get('f_score', 0))
                secondary_metrics.append(record.get('accuracy', 0))
            elif args.task_name == 'classification':
                primary_metrics.append(record.get('accuracy', 0))
                secondary_metrics.append(record.get('accuracy', 0))  # 分类任务只有准确率
            
            # 按实验分组统计
            exp_timestamp = record['experiment_timestamp']
            if exp_timestamp not in experiment_groups:
                experiment_groups[exp_timestamp] = []
            experiment_groups[exp_timestamp].append(record)
    
    # (8) 添加整体统计汇总行
    if primary_metrics and secondary_metrics:
        summary_row = params_dict.copy()
        base_summary = {
            'iteration': 'SUMMARY',
            'total_iterations': len(all_data),
            'total_experiments': len(experiment_groups),
            'experiment_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'train_time': '',
            'final_train_loss': '',
            'final_vali_loss': '',
            'epochs_trained': ''
        }
        
        # 根据任务类型添加统计信息
        if args.task_name == 'long_term_forecast':
            base_summary.update({
                'mse': np.mean(primary_metrics),
                'mae': np.mean(secondary_metrics),
                'mse_std': np.std(primary_metrics),
                'mae_std': np.std(secondary_metrics),
                'rmse': '',
                'mape': '',
                'mspe': '',
                'dtw': ''
            })
        elif args.task_name == 'imputation':
            base_summary.update({
                'mse': np.mean(primary_metrics),
                'mae': np.mean(secondary_metrics),
                'mse_std': np.std(primary_metrics),
                'mae_std': np.std(secondary_metrics),
                'rmse': '',
                'mape': '',
                'mspe': '',
                'mask_rate': ''
            })
        elif args.task_name == 'short_term_forecast':
            base_summary.update({
                'smape': np.mean(primary_metrics),
                'owa': np.mean(secondary_metrics),
                'smape_std': np.std(primary_metrics),
                'owa_std': np.std(secondary_metrics),
                'mape': '',
                'mase': '',
                'seasonal_patterns': '',
                'status': ''
            })
        elif args.task_name == 'anomaly_detection':
            base_summary.update({
                'f_score': np.mean(primary_metrics),
                'accuracy': np.mean(secondary_metrics),
                'f_score_std': np.std(primary_metrics),
                'accuracy_std': np.std(secondary_metrics),
                'precision': '',
                'recall': '',
                'threshold': '',
                'anomaly_ratio': ''
            })
        elif args.task_name == 'classification':
            base_summary.update({
                'accuracy': np.mean(primary_metrics),
                'accuracy_std': np.std(primary_metrics),
                'num_classes': '',
                'total_samples': ''
            })
        
        summary_row.update(base_summary)
        all_data.append(summary_row)
    
    # (9) 保存到CSV文件
    df = pd.DataFrame(all_data)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    print(f"{args.task_name}任务实验结果已保存到: {filepath}")
    print(f"当前文件包含 {len(experiment_groups)} 个实验组，共 {len([d for d in all_data if isinstance(d['iteration'], int)])} 次迭代")
    
    return filepath

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
        # (1) 初始化结果收集容器 - 用于存储每次迭代的详细结果
        results_list = []  # 存储每次迭代的详细结果
        all_mses = []      # 存储所有MSE值用于最终统计
        all_maes = []      # 存储所有MAE值用于最终统计
        
        for ii in range(args.itr):
            # (2) 记录当前迭代开始时间
            import time
            iteration_start_time = time.time()
            
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
            # (3) 执行训练并记录训练开始时间
            train_start_time = time.time()
            trained_model = exp.train(setting)  # 开始训练
            train_end_time = time.time()
            train_time = train_end_time - train_start_time

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            # (4) 执行测试并收集结果指标
            test_results = exp.test(setting)  # 训练后测试，现在返回指标字典
            
            # (5) 收集当前迭代的完整结果数据
            if test_results:  # 确保test_results不为空
                # 基础结果数据
                iteration_result = {
                    'train_time': train_time,
                    'final_train_loss': 0,  # 这需要从训练过程中获取
                    'final_vali_loss': 0,   # 这需要从训练过程中获取
                    'epochs_trained': args.train_epochs,  # 假设训练了全部轮数
                }
                
                # 添加所有返回的指标
                iteration_result.update(test_results)
                
                results_list.append(iteration_result)
                
                # 根据任务类型收集主要指标用于统计
                if args.task_name == 'long_term_forecast' or args.task_name == 'imputation':
                    all_mses.append(iteration_result.get('mse', 0))
                    all_maes.append(iteration_result.get('mae', 0))
                    main_metric = iteration_result.get('mse', 0)
                    secondary_metric = iteration_result.get('mae', 0)
                    metric_names = ('MSE', 'MAE')
                elif args.task_name == 'short_term_forecast':
                    all_mses.append(iteration_result.get('smape', 0))
                    all_maes.append(iteration_result.get('owa', 0))
                    main_metric = iteration_result.get('smape', 0)
                    secondary_metric = iteration_result.get('owa', 0)
                    metric_names = ('SMAPE', 'OWA')
                elif args.task_name == 'anomaly_detection':
                    all_mses.append(iteration_result.get('f_score', 0))
                    all_maes.append(iteration_result.get('accuracy', 0))
                    main_metric = iteration_result.get('f_score', 0)
                    secondary_metric = iteration_result.get('accuracy', 0)
                    metric_names = ('F-Score', 'Accuracy')
                elif args.task_name == 'classification':
                    all_mses.append(iteration_result.get('accuracy', 0))
                    all_maes.append(iteration_result.get('accuracy', 0))
                    main_metric = iteration_result.get('accuracy', 0)
                    secondary_metric = iteration_result.get('accuracy', 0)
                    metric_names = ('Accuracy', 'Accuracy')
                
                # (6) 打印当前迭代的关键指标
                iteration_end_time = time.time()
                total_iteration_time = iteration_end_time - iteration_start_time
                print(f"迭代 {ii+1}/{args.itr} 完成:")
                print(f"  {metric_names[0]}: {main_metric:.6f}, {metric_names[1]}: {secondary_metric:.6f}")
                print(f"  训练时间: {train_time:.2f}秒, 总时间: {total_iteration_time:.2f}秒")
            
            # 清理显存
            if args.gpu_type == "mps":
                torch.backends.mps.empty_cache()
            elif args.gpu_type == "cuda":
                torch.cuda.empty_cache()
        
        # (7) 所有迭代完成后，计算最终统计信息并保存到CSV
        if results_list:  # 确保有结果数据
            # 计算整个实验的统计信息
            primary_values = np.array(all_mses)
            secondary_values = np.array(all_maes)
            
            # 根据任务类型创建统计信息
            if args.task_name == 'long_term_forecast' or args.task_name == 'imputation':
                final_stats = {
                    'mse_mean': np.mean(primary_values),
                    'mse_std': np.std(primary_values),
                    'mae_mean': np.mean(secondary_values),
                    'mae_std': np.std(secondary_values)
                }
                stat_labels = ('MSE', 'MAE')
            elif args.task_name == 'short_term_forecast':
                final_stats = {
                    'smape_mean': np.mean(primary_values),
                    'smape_std': np.std(primary_values),
                    'owa_mean': np.mean(secondary_values),
                    'owa_std': np.std(secondary_values)
                }
                stat_labels = ('SMAPE', 'OWA')
            elif args.task_name == 'anomaly_detection':
                final_stats = {
                    'f_score_mean': np.mean(primary_values),
                    'f_score_std': np.std(primary_values),
                    'accuracy_mean': np.mean(secondary_values),
                    'accuracy_std': np.std(secondary_values)
                }
                stat_labels = ('F-Score', 'Accuracy')
            elif args.task_name == 'classification':
                final_stats = {
                    'accuracy_mean': np.mean(primary_values),
                    'accuracy_std': np.std(primary_values)
                }
                stat_labels = ('Accuracy',)
            
            print("\n" + "="*60)
            print("实验统计结果:")
            print(f"{stat_labels[0]}: {np.mean(primary_values):.6f} ± {np.std(primary_values):.6f}")
            if len(stat_labels) > 1:
                print(f"{stat_labels[1]}: {np.mean(secondary_values):.6f} ± {np.std(secondary_values):.6f}")
            print("="*60)
            
            # 一次性保存整个实验的所有结果到CSV文件
            csv_filename = save_results_to_csv(args, results_list, final_stats)
            print(f"\n完整的实验结果已保存到CSV文件: {csv_filename}")
        else:
            print("警告: 没有收集到任何实验结果数据")
    else:
        # 只做测试模式 - 加载已训练模型进行测试并保存结果
        import time
        test_start_time = time.time()
        
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
        # 执行测试并收集结果
        test_results = exp.test(setting, test=1)  # 只测试，现在返回指标字典
        test_end_time = time.time()
        test_time = test_end_time - test_start_time
        
        # 收集测试结果用于CSV保存
        if test_results:  # 确保test_results不为空
            # 基础测试结果
            base_result = {
                'train_time': 0,  # 测试模式下没有训练时间
                'final_train_loss': 0,  # 测试模式下没有训练损失
                'final_vali_loss': 0,   # 测试模式下没有验证损失
                'epochs_trained': 0,    # 测试模式下没有训练
                'test_time': test_time
            }
            
            # 添加所有测试返回的指标
            base_result.update(test_results)
            results_list = [base_result]
            
            # 根据任务类型计算统计信息（单次测试）
            if args.task_name == 'long_term_forecast' or args.task_name == 'imputation':
                main_metric = test_results.get('mse', 0)
                secondary_metric = test_results.get('mae', 0)
                metric_labels = ['MSE', 'MAE', 'RMSE']
                metric_values = [main_metric, secondary_metric, test_results.get('rmse', 0)]
                final_stats = {'mse_mean': main_metric, 'mse_std': 0, 'mae_mean': secondary_metric, 'mae_std': 0}
            elif args.task_name == 'short_term_forecast':
                main_metric = test_results.get('smape', 0)
                secondary_metric = test_results.get('owa', 0)
                metric_labels = ['SMAPE', 'OWA', 'MAPE', 'MASE']
                metric_values = [main_metric, secondary_metric, test_results.get('mape', 0), test_results.get('mase', 0)]
                final_stats = {'smape_mean': main_metric, 'smape_std': 0, 'owa_mean': secondary_metric, 'owa_std': 0}
            elif args.task_name == 'anomaly_detection':
                main_metric = test_results.get('f_score', 0)
                secondary_metric = test_results.get('accuracy', 0)
                metric_labels = ['F-Score', 'Accuracy', 'Precision', 'Recall']
                metric_values = [main_metric, secondary_metric, test_results.get('precision', 0), test_results.get('recall', 0)]
                final_stats = {'f_score_mean': main_metric, 'f_score_std': 0, 'accuracy_mean': secondary_metric, 'accuracy_std': 0}
            elif args.task_name == 'classification':
                main_metric = test_results.get('accuracy', 0)
                metric_labels = ['Accuracy']
                metric_values = [main_metric]
                final_stats = {'accuracy_mean': main_metric, 'accuracy_std': 0}
            
            print("\n" + "="*60)
            print("测试结果:")
            for label, value in zip(metric_labels, metric_values):
                print(f"{label}: {value:.6f}")
            print(f"测试时间: {test_time:.2f}秒")
            print("="*60)
            
            # 保存测试结果到CSV（使用任务特定的测试结果文件名）
            task_test_file_mapping = {
                'long_term_forecast': 'long_term_forecasting_test_results.csv',
                'short_term_forecast': 'short_term_forecasting_test_results.csv',
                'anomaly_detection': 'anomaly_detection_test_results.csv',
                'classification': 'classification_test_results.csv',
                'imputation': 'imputation_test_results.csv'
            }
            test_filename = task_test_file_mapping.get(args.task_name, 'unknown_task_test_results.csv')
            csv_filename = save_results_to_csv(args, results_list, final_stats, filename=test_filename)
            print(f"\n{args.task_name}任务测试结果已保存到CSV文件: {csv_filename}")
        else:
            print("警告: 测试未返回有效结果")
        
        # 清理显存
        if args.gpu_type == "mps":
            torch.backends.mps.empty_cache()
        elif args.gpu_type == "cuda":
            torch.cuda.empty_cache()