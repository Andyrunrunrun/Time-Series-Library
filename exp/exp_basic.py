import os
import torch
# 从 models 包中导入所有支持的时间序列模型类
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer


# 定义实验基础类，所有实验类都应继承自该类
class Exp_Basic(object):
    def __init__(self, args):
        """
        初始化实验基础类，设置模型字典、设备和模型实例。
        :param args: 命令行参数或配置对象，包含模型名称、设备信息等
        """
        self.args = args
        # 支持的模型名称与对应类的映射字典
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer
        }
        # 如果选择的模型是 Mamba，需要动态导入并添加到模型字典
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        # 获取当前设备（GPU/MPS/CPU）
        self.device = self._acquire_device()
        # 构建模型并移动到指定设备
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """
        构建模型的方法，需在子类中实现。
        :return: 构建好的模型实例
        """
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        根据参数自动选择设备（GPU/MPS/CPU）。
        :return: torch.device 对象
        """
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            # 设置可见的 CUDA 设备
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            # Apple M1/M2 芯片的 GPU 支持
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            # 默认使用 CPU
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        """
        获取数据的方法，需在子类中实现。
        """
        pass

    def vali(self):
        """
        验证模型的方法，需在子类中实现。
        """
        pass

    def train(self):
        """
        训练模型的方法，需在子类中实现。
        """
        pass

    def test(self):
        """
        测试模型的方法，需在子类中实现。
        """
        pass
