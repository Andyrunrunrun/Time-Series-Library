import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    """完整注意力机制的实现
    这是transformer中标准的缩放点积注意力机制(Scaled Dot-Product Attention)的实现
    计算公式为: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """初始化函数
        Args:
            mask_flag (bool): 是否使用掩码，用于因果注意力机制
            factor (int): 缩放因子，用于ProbAttention中，这里未使用
            scale (float): 缩放因子，若为None则使用1/sqrt(d_k)
            attention_dropout (float): 注意力权重的dropout率
            output_attention (bool): 是否输出注意力权重矩阵
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """前向传播函数
        Args:
            queries: 查询矩阵 [Batch, Length_Q, Head, Embedding_dim]
            keys: 键矩阵 [Batch, Length_K, Head, Embedding_dim]
            values: 值矩阵 [Batch, Length_V, Head, Dimension_V]
            attn_mask: 注意力掩码
            tau: 温度参数(未使用)
            delta: 位置偏移参数(未使用)
        Returns:
            V: 注意力计算结果
            A: 注意力权重矩阵(如果output_attention=True)
        """
        # 获取输入张量的形状
        B, L, H, E = queries.shape  # Batch, Length_Q, Head, Embedding_dim
        _, S, _, D = values.shape   # Batch, Length_V, Head, Dimension_V
        
        # 计算缩放因子，如果未指定则使用1/sqrt(E)
        scale = self.scale or 1. / sqrt(E)

        # 计算注意力分数 (Q×K^T)
        # einsum进行批量矩阵乘法，得到形状为[B,H,L,S]的张量
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 如果使用掩码
        if self.mask_flag:
            if attn_mask is None:
                # 创建三角因果掩码，用于确保当前位置只能注意到过去的位置
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            
            # 将掩码位置的值设为负无穷，使softmax后的权重为0
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 应用softmax和dropout得到注意力权重
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # 计算最终的注意力输出 (A×V)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # 根据output_attention决定是否返回注意力权重矩阵
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """多头注意力机制层的实现
    这个类封装了完整的多头注意力机制，包括投影变换和注意力计算
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        """初始化函数
        Args:
            attention: 注意力机制的具体实现（如FullAttention、ProbAttention等）
            d_model: 输入的特征维度
            n_heads: 注意力头的数量
            d_keys: 键向量的维度，默认为d_model/n_heads
            d_values: 值向量的维度，默认为d_model/n_heads
        """
        super(AttentionLayer, self).__init__()

        # 如果没有指定键和值的维度，则将输入维度平均分配给每个注意力头
        d_keys = d_keys or (d_model // n_heads)    # 每个头的键维度
        d_values = d_values or (d_model // n_heads) # 每个头的值维度

        # 保存注意力机制的具体实现（如FullAttention）
        self.inner_attention = attention
        
        # 创建线性变换层，用于生成查询、键、值向量
        # 输入维度为d_model，输出维度为d_keys/d_values * n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)   # Q投影
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)     # K投影
        self.value_projection = nn.Linear(d_model, d_values * n_heads) # V投影
        
        # 输出投影，将多头的结果合并回原始维度
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        # 保存注意力头数量
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """前向传播函数
        Args:
            queries: 查询张量 [Batch, Length_Q, d_model]
            keys: 键张量 [Batch, Length_K, d_model]
            values: 值张量 [Batch, Length_V, d_model]
            attn_mask: 注意力掩码
            tau: 温度参数（可选）
            delta: 位置偏移参数（可选）
        Returns:
            out: 注意力机制的输出 [Batch, Length_Q, d_model]
            attn: 注意力权重
        """
        # 获取批次大小和序列长度
        B, L, _ = queries.shape  # B:批次大小，L:查询序列长度
        _, S, _ = keys.shape     # S:键序列长度
        H = self.n_heads        # H:注意力头数量

        # 对查询、键、值进行线性变换并重塑维度
        # 从[B, L, d_model]变换到[B, L, H, d_k/v]
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 调用具体的注意力机制实现（如FullAttention）
        out, attn = self.inner_attention(
            queries,    # [B, L, H, d_k]
            keys,      # [B, S, H, d_k]
            values,    # [B, S, H, d_v]
            attn_mask, # 注意力掩码
            tau=tau,   # 温度参数
            delta=delta # 位置偏移参数
        )
        
        # 将多头注意力的结果重塑回原始维度
        # 从[B, L, H, d_v]变换到[B, L, H*d_v]
        out = out.view(B, L, -1)

        # 通过输出投影层，将结果映射回原始维度
        # 从[B, L, H*d_v]变换到[B, L, d_model]
        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
