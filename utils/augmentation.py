import numpy as np
from tqdm import tqdm

def jitter(x, sigma=0.03):
    """
    通过添加高斯噪声对输入的时间序列数据进行抖动。
    这是一种常见的数据增强技术，用以提高模型的鲁棒性。
    参考文献: https://arxiv.org/pdf/1706.00527.pdf

    Args:
        x (np.ndarray): 输入的时间序列数据。
                        形状应为 (batch_size, sequence_length, num_features)。
        sigma (float): 要添加的高斯噪声的标准差。

    Returns:
        np.ndarray: 增加了噪声的增强时间序列数据。
    """
    # 添加从均值为0，标准差为sigma的正态分布中抽取的随机噪声
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    """
    通过一个随机因子缩放输入时间序列数据的幅度。
    缩放因子从高斯分布中抽取。
    参考文献: https://arxiv.org/pdf/1706.00527.pdf

    Args:
        x (np.ndarray): 输入的时间序列数据。
                        形状应为 (batch_size, sequence_length, num_features)。
        sigma (float): 缩放因子高斯分布的标准差（均值为1）。

    Returns:
        np.ndarray: 经过缩放的时间序列数据。
    """
    # 为每个样本和特征生成一个随机缩放因子
    # 因子从均值为1，标准差为sigma的正态分布中抽取。
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    # 将缩放因子应用于每个时间序列。
    # np.newaxis 用于在时间步上正确地广播因子。
    return np.multiply(x, factor[:, np.newaxis, :])

def rotation(x):
    """
    对输入数据应用随机旋转。
    通过随机翻转每个特征的符号并排列特征轴来实现。

    Args:
        x (np.ndarray): 输入的时间序列数据。
                        形状应为 (batch_size, sequence_length, num_features)。

    Returns:
        np.ndarray: 经过旋转的时间序列数据。
    """
    x = np.array(x)
    # 为每个样本和特征随机选择是否翻转符号（-1）或不翻转（1）。
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    # 创建一个表示特征轴的数组。
    rotate_axis = np.arange(x.shape[2])
    # 随机打乱特征轴。
    np.random.shuffle(rotate_axis)
    # 应用符号翻转和轴排列。
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]

def permutation(x, max_segments=5, seg_mode="equal"):
    """
    对时间序列的段进行排列。序列被分割成若干段，然后这些段被随机打乱。

    Args:
        x (np.ndarray): 输入的时间序列数据。
                        形状应为 (batch_size, sequence_length, num_features)。
        max_segments (int): 将时间序列分割成的最大段数。
        seg_mode (str): 如何创建段。"equal"表示等大段，"random"表示随机大小的段。

    Returns:
        np.ndarray: 段被排列过的时间序列数据。
    """
    orig_steps = np.arange(x.shape[1])
    
    # 确定批次中每个样本的段数。
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                # 为段选择随机的分割点。
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                # 分割成等大的段。
                splits = np.array_split(orig_steps, num_segs[i])
            
            # 排列段并将新顺序应用于原始数据。
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            # 如果只有一个段，则不进行任何更改。
            ret[i] = pat
    return ret

def magnitude_warp(x, sigma=0.2, knot=4):
    """
    使用平滑曲线扭曲时间序列的幅度。
    将三次样条拟合到随机锚点，并使用此样条随时间缩放序列的幅度。

    Args:
        x (np.ndarray): 输入的时间序列数据。
                        形状应为 (batch_size, sequence_length, num_features)。
        sigma (float): 样条节点随机值的标准差。
        knot (int): 三次样条的节点数。

    Returns:
        np.ndarray: 幅度扭曲的时间序列数据。
    """
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    # 为样条的节点生成随机扭曲因子。
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    # 定义节点的时间点，均匀分布在序列长度上。
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        # 为每个特征维度创建一个三次样条。
        warper = np.array([CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        # 通过逐元素相乘应用扭曲。
        ret[i] = pat * warper

    return ret

def time_warp(x, sigma=0.2, knot=4):
    """
    使用平滑曲线扭曲时间序列的时间维度。
    使用三次样条扭曲时间步，有效地加速或减慢序列的某些部分。

    Args:
        x (np.ndarray): 输入的时间序列数据。
                        形状应为 (batch_size, sequence_length, num_features)。
        sigma (float): 用于生成随机时间扭曲的标准差。
        knot (int): 三次样条的节点数。

    Returns:
        np.ndarray: 时间扭曲的时间序列数据。
    """
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    # 为节点生成随机扭曲因子。
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    # 定义节点的时间点。
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            # 创建一个将原始时间步映射到扭曲时间步的样条。
            time_warp_spline = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])
            time_warp = time_warp_spline(orig_steps)
            
            # 缩放扭曲的时间以确保其覆盖原始时间范围。
            scale = (x.shape[1] - 1) / time_warp[-1]
            
            # 将原始模式插值到新的扭曲时间步上。
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret

def window_slice(x, reduce_ratio=0.9):
    """
    通过取时间序列的随机切片并将其拉伸到原始长度来执行窗口切片。
    参考文献: https://halshs.archives-ouvertes.fr/halshs-01357973/document

    Args:
        x (np.ndarray): 输入的时间序列数据。
                        形状应为 (batch_size, sequence_length, num_features)。
        reduce_ratio (float): 要切片的原始长度的比例。例如，0.9表示
                              切片将是原始长度的90%。

    Returns:
        np.ndarray: 经过窗口切片和调整大小的时间序列数据。
    """
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x # 如果目标长度不小于原始长度，则不进行切片。
        
    # 为每个样本确定切片的随机起始点。
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            # 取切片。
            sliced_pat = pat[starts[i]:ends[i], dim]
            # 将切片插值回原始序列长度。
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), sliced_pat).T
    return ret

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    """
    通过在时间序列中随机选择一个窗口并加速或减速它来执行窗口扭曲。
    参考文献: https://halshs.archives-ouvertes.fr/halshs-01357973/document

    Args:
        x (np.ndarray): 输入的时间序列数据。
                        形状应为 (batch_size, sequence_length, num_features)。
        window_ratio (float): 选择作为窗口的时间序列的比例。
        scales (list): 窗口扭曲的可能缩放因子列表，
                       例如, [0.5, 2.0] 表示加速2倍或减速0.5倍。

    Returns:
        np.ndarray: 经过窗口扭曲的时间序列数据。
    """
    # 为批次中的每个样本选择一个扭曲比例。
    warp_scales = np.random.choice(scales, x.shape[0])
    # 计算要扭曲的窗口的大小。
    warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    # 为窗口选择随机的起始点。
    window_starts = np.random.randint(low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            # 将序列分成三部分：窗口前、窗口中和窗口后。
            start_seg = pat[:window_starts[i], dim]
            # 通过将其插值到新长度来扭曲窗口段。
            window_seg = np.interp(np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i], dim])
            end_seg = pat[window_ends[i]:, dim]
            
            # 连接各部分并插值回原始长度。
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i, :, dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1] - 1., num=warped.size), warped).T
    return ret

def spawner(x, labels, sigma=0.05, verbose=0):
    """
    通过基于动态时间规整（DTW）路径平均两个相同类别的现有序列来生成新的时间序列。
    参考文献: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/

    Args:
        x (np.ndarray): 输入的时间序列数据。
        labels (np.ndarray): 输入数据的标签。
        sigma (float): 最后添加的抖动的标准差。
        verbose (int): 详细级别。

    Returns:
        np.ndarray: 增强后的数据。
    """
    import utils.dtw as dtw
    # 为每个序列选择一个随机的分割点。
    random_points = np.random.randint(low=1, high=x.shape[1] - 1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        # 找到相同类别的其他样本。
        choices = np.delete(np.arange(x.shape[0]), i)
        choices = np.where(l[choices] == l[i])[0]
        
        if choices.size > 0:     
            # 从相同类别中随机选择一个样本。
            random_sample = x[np.random.choice(choices)]
            
            # 计算序列两部分的DTW路径，在随机点分割。
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            
            # 合并路径。
            combined = np.concatenate((np.vstack(path1), np.vstack(path2 + random_points[i])), axis=1)
            
            # 通过沿DTW路径平均来创建新样本。
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                # 将平均后的序列插值回原始长度。
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=mean.shape[0]), mean[:, dim]).T
        else:
            # 如果不存在相同类别的其他样本，则不进行增强。
            ret[i, :] = pat
            
    # 对最终结果添加抖动。
    return jitter(ret, sigma=sigma)

def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    """
    执行加权DTW重心平均（WDBA）。
    它通过对来自相同类别的一组序列进行加权平均来创建一个新序列，其中权重由DTW距离确定。
    参考文献: https://ieeexplore.ieee.org/document/8215569

    Args:
        x (np.ndarray): 输入的时间序列数据。
        labels (np.ndarray): 输入数据的标签。
        batch_size (int): 用于平均的类内样本数。
        slope_constraint (str): DTW斜率约束（"symmetric"或"asymmetric"）。
        use_window (bool): 是否使用窗口进行DTW计算。
        verbose (int): 详细级别。

    Returns:
        np.ndarray: 增强后的数据。
    """
    import utils.dtw as dtw
    x = np.array(x)
    
    window = np.ceil(x.shape[1] / 10.).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        
    ret = np.zeros_like(x)
    for i in range(ret.shape[0]):
        # 找到相同类别的其他样本。
        choices = np.where(l == l[i])[0]
        if choices.size > 0:        
            # 随机选择一批类内模式。
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]
            
            # 计算所选原型的成对DTW距离矩阵。
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p != s:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                        
            # 找到中心点（到其他样本的距离总和最小的样本）。
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]
            
            # 开始加权DBA。
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    # 中心点本身的权重为1。
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums) 
                else:
                    # 对于其他模式，计算到中心点的DTW路径。
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    # 根据路径扭曲模式。
                    warped = random_prototypes[nid, path[1]]
                    # 权重随DTW距离呈指数衰减。
                    weight = np.exp(np.log(0.5) * dtw_value / dtw_matrix[medoid_id, nearest_order[1]])
                    # 将加权的、扭曲的模式添加到平均值中。
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight 
            
            # 通过权重总和进行归一化，以获得最终的平均模式。
            ret[i, :] = average_pattern / weighted_sums[:, np.newaxis]
        else:
            ret[i, :] = x[i]
    return ret

# 提出的增强方法

def random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal", verbose=0):
    """
    使用从DTW派生的路径，将时间序列朝向一个随机选择的相同类别的原型进行扭曲。

    Args:
        x (np.ndarray): 输入的时间序列数据。
        labels (np.ndarray): 输入数据的标签。
        slope_constraint (str): DTW斜率约束。
        use_window (bool): 是否使用窗口进行DTW。
        dtw_type (str): 使用的DTW类型（"normal"或"shape"）。
        verbose (int): 详细级别。

    Returns:
        np.ndarray: 增强后的数据。
    """
    import utils.dtw as dtw
    
    window = np.ceil(x.shape[1] / 10.).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        # 找到相同类别的其他样本。
        choices = np.delete(np.arange(x.shape[0]), i)
        choices = np.where(l[choices] == l[i])[0]
        
        if choices.size > 0:        
            # 随机选择一个类内原型。
            random_prototype = x[np.random.choice(choices)]
            
            # 计算原型和当前模式之间的DTW路径。
            if dtw_type == "shape":
                path = dtw.shape_dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                            
            # 基于DTW路径对模式进行时间扭曲。
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                # 将扭曲后的模式插值回原始长度。
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped.shape[0]), warped[:, dim]).T
        else:
            ret[i, :] = pat
    return ret

def random_guided_warp_shape(x, labels, slope_constraint="symmetric", use_window=True):
    """
    一个专门使用shapeDTW的`random_guided_warp`的包装器。
    """
    return random_guided_warp(x, labels, slope_constraint, use_window, dtw_type="shape")

def discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="normal", use_variable_slice=True, verbose=0):
    """
    执行判别性引导扭曲。它将一个样本朝向一个类内原型进行扭曲，
    该原型被选择为与其他类别最大程度地区分。
    可选择在最后应用可变窗口切片。

    Args:
        x (np.ndarray): 输入的时间序列数据。
        labels (np.ndarray): 输入数据的标签。
        batch_size (int): 要考虑的原型数量。
        slope_constraint (str): DTW斜率约束。
        use_window (bool): 是否使用窗口进行DTW。
        dtw_type (str): 使用的DTW类型（"normal"或"shape"）。
        use_variable_slice (bool): 是否在扭曲后应用可变窗口切片。
        verbose (int): 详细级别。

    Returns:
        np.ndarray: 增强后的数据。
    """
    import utils.dtw as dtw
    
    window = np.ceil(x.shape[1] / 10.).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    # 为正（类内）和负（类间）样本分割批次大小。
    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)
        
    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(x):
        choices = np.delete(np.arange(x.shape[0]), i)
        
        # 获取正（相同类别）和负（不同类别）样本。
        positive = np.where(l[choices] == l[i])[0]
        negative = np.where(l[choices] != l[i])[0]
        
        if positive.size > 0 and negative.size > 0:
            # 从正集和负集中选择随机原型。
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]
                        
            # 找到最具判别性的正原型。
            # 这是平均而言，与其他正例“最接近”
            # 且与负例“最远”的那个。
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            dtw_func = dtw.shape_dtw if dtw_type == "shape" else dtw.dtw
            
            for p, pos_prot in enumerate(positive_prototypes):
                for ps, pos_samp in enumerate(positive_prototypes):
                    if p != ps:
                        pos_aves[p] += (1. / (pos_k - 1.)) * dtw_func(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                for ns, neg_samp in enumerate(negative_prototypes):
                    neg_aves[p] += (1. / neg_k) * dtw_func(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
            
            # 选择最大化（到负例的距离 - 到正例的距离）的原型。
            selected_id = np.argmax(neg_aves - pos_aves)
            path = dtw_func(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                   
            # 使用从所选原型获得的路径对原始模式进行时间扭曲。
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps - warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped.shape[0]), warped[:, dim]).T
        else:
            ret[i, :] = pat
            warp_amount[i] = 0.
            
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # 如果没有发生扭曲，则应用标准的窗口切片。
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            # 应用与所施加的扭曲量成比例的比例的窗口切片。
            for i, pat in enumerate(ret):
                reduce_ratio = 0.9 + 0.1 * warp_amount[i] / max_warp
                ret[i] = window_slice(pat[np.newaxis, :, :], reduce_ratio=reduce_ratio)[0]
    return ret

def discriminative_guided_warp_shape(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    """
    一个专门使用shapeDTW的`discriminative_guided_warp`的包装器。
    """
    return discriminative_guided_warp(x, labels, batch_size, slope_constraint, use_window, dtw_type="shape")


def run_augmentation(x, y, args):
    """
    对数据集多次运行一系列增强。

    Args:
        x (np.ndarray): 输入数据。
        y (np.ndarray): 输入标签。
        args (Namespace): 一个包含增强参数的命名空间对象，
                          包括`augmentation_ratio`和每种类型的标志。

    Returns:
        tuple: 一个包含以下内容的元组：
            - np.ndarray: 增强后的数据（原始+新增）。
            - np.ndarray: 相应的标签。
            - str: 描述所应用增强的标签。
    """
    print("Augmenting %s" % args.data)
    np.random.seed(args.seed)
    x_aug = x
    y_aug = y
    if args.augmentation_ratio > 0:
        augmentation_tags = "%d" % args.augmentation_ratio
        # 根据augmentation_ratio指定的次数应用增强
        for n in range(args.augmentation_ratio):
            x_temp, augmentation_tags_round = augment(x, y, args)
            x_aug = np.append(x_aug, x_temp, axis=0)
            y_aug = np.append(y_aug, y, axis=0) # 为增强数据附加原始标签
            print("Round %d: %s done" % (n, augmentation_tags_round))
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = args.extra_tag
    return x_aug, y_aug, augmentation_tags

def run_augmentation_single(x, y, args):
    """
    对单个时间序列或一个批次运行增强，处理维度变化。

    Args:
        x (np.ndarray): 输入数据，可以是2D或3D。
        y (np.ndarray): 输入标签。
        args (Namespace): 一个包含增强参数的命名空间对象。

    Returns:
        tuple: 一个包含以下内容的元组：
            - np.ndarray: 增强后的数据。
            - np.ndarray: 相应的标签。
            - str: 描述所应用增强的标签。
    """
    np.random.seed(args.seed)

    x_aug = x
    y_aug = y

    if len(x.shape) < 3:
        # 如果输入是单个序列（长度，特征），则添加一个批次维度。
        x_input = x[np.newaxis, :]
    elif len(x.shape) == 3:
        # 如果输入已经是批次，则直接使用。
        x_input = x
    else:
        raise ValueError("输入必须是 (batch_size, sequence_length, num_channels) 维度")

    if args.augmentation_ratio > 0:
        augmentation_tags = "%d" % args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            # 注意：这会在每个循环中覆盖x_aug，如果augmentation_ratio > 1，这可能不是预期的行为。
            # 它似乎旨在应用一组增强。
            x_aug, augmentation_tags_round = augment(x_input, y, args)
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = args.extra_tag

    if len(x.shape) < 3:
        # 如果原始输入是单个序列，则移除批次维度。
        x_aug = x_aug.squeeze(0)
    return x_aug, y_aug, augmentation_tags


def augment(x, y, args):
    """
    根据args对象中的布尔标志应用一组特定的增强。

    Args:
        x (np.ndarray): 输入数据。
        y (np.ndarray): 输入标签。
        args (Namespace): 一个带有每个增强的布尔标志的命名空间对象。

    Returns:
        tuple: 一个包含以下内容的元组：
            - np.ndarray: 增强后的数据。
            - str: 描述所应用增强的标签。
    """
    import utils.augmentation as aug
    augmentation_tags = ""
    # 将指定的增强链接在一起
    if args.jitter:
        x = aug.jitter(x)
        augmentation_tags += "_jitter"
    if args.scaling:
        x = aug.scaling(x)
        augmentation_tags += "_scaling"
    if args.rotation:
        x = aug.rotation(x)
        augmentation_tags += "_rotation"
    if args.permutation:
        x = aug.permutation(x)
        augmentation_tags += "_permutation"
    if args.randompermutation:
        x = aug.permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    if args.magwarp:
        x = aug.magnitude_warp(x)
        augmentation_tags += "_magwarp"
    if args.timewarp:
        x = aug.time_warp(x)
        augmentation_tags += "_timewarp"
    if args.windowslice:
        x = aug.window_slice(x)
        augmentation_tags += "_windowslice"
    if args.windowwarp:
        x = aug.window_warp(x)
        augmentation_tags += "_windowwarp"
    if args.spawner:
        x = aug.spawner(x, y)
        augmentation_tags += "_spawner"
    if args.dtwwarp:
        x = aug.random_guided_warp(x, y)
        augmentation_tags += "_rgw"
    if args.shapedtwwarp:
        x = aug.random_guided_warp_shape(x, y)
        augmentation_tags += "_rgws"
    if args.wdba:
        x = aug.wdba(x, y)
        augmentation_tags += "_wdba"
    if args.discdtw:
        x = aug.discriminative_guided_warp(x, y)
        augmentation_tags += "_dgw"
    if args.discsdtw:
        x = aug.discriminative_guided_warp_shape(x, y)
        augmentation_tags += "_dgws"
    return x, augmentation_tags
