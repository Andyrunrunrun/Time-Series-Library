import torch


class TriangularCausalMask():
    """
    三角形因果掩码类
    用于实现自注意力机制中的因果掩码，确保当前时间步只能访问过去的信息
    
    参数:
    - B: batch size（批次大小）
    - L: sequence length（序列长度）
    - device: 运行设备，默认为CPU
    """
    def __init__(self, B, L, device="cpu"):
        # 定义掩码形状 [batch_size, 1, seq_len, seq_len]
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # 创建上三角矩阵（对角线上移一位），用于屏蔽未来信息
            # triu 函数创建上三角矩阵，diagonal=1 表示对角线上移一位
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """
        返回掩码张量的属性装饰器
        """
        return self._mask


class ProbMask():
    """
    概率掩码类
    用于实现基于概率的注意力掩码，通常用在稀疏注意力机制中
    
    参数:
    - B: batch size（批次大小）
    - H: number of heads（注意力头数）
    - L: sequence length（序列长度）
    - index: 索引张量
    - scores: 注意力分数
    - device: 运行设备，默认为CPU
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # 创建上三角掩码矩阵
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        # 扩展掩码维度以匹配批次和多头注意力的形状
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        # 根据提供的索引创建指示器掩码
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        # 调整掩码形状以匹配分数张量
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """
        返回掩码张量的属性装饰器
        """
        return self._mask
