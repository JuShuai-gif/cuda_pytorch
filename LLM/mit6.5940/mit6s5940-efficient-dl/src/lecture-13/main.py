#!/usr/bin/env python3
"""
第 13 讲：LLM 部署模拟
======================================

本脚本在 CPU 上模拟 LLM 部署的关键优化技术：

  - 仅权重量化：对线性层权重进行分组 4-bit（AWQ 风格）量化，
    测量在合成文本上的困惑度退化。
  - 模型大小比较：FP32 vs FP16 vs INT4，展示存储缩减
    比例和元数据开销。
  - KV 缓存量化：在不同上下文长度下比较 FP16 vs INT8 KV 缓存存储，
    展示 2 倍的内存节省。
  - FlashAttention 概念：大量内联注释解释分块（tiling）算法
    （无实际实现，仅概念讲解）。

所有计算均在 CPU 上执行。依赖仅限于 torch、numpy 和
Python 标准库（math）。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 第 0 部分：配置
# =============================================================================


@dataclass
class GPTConfig:
    """本模拟中使用的小型 GPT 模型的超参数。"""

    vocab_size: int = 1000  # 合成词汇表大小
    d_model: int = 256  # 隐藏维度
    n_heads: int = 4  # 注意力头数
    n_layers: int = 3  # Transformer 层数
    d_ff: int = 1024  # 前馈网络内部维度
    max_seq_len: int = 256  # 最大序列长度
    dropout: float = 0.0  # Dropout 率（0.0 用于确定性评估）


# =============================================================================
# 第 1 部分：小型 GPT 模型定义
# =============================================================================


class MultiHeadAttention(nn.Module):
    """标准多头缩放点积注意力。

    将隐藏维度拆分为 ``n_heads`` 个独立的头，计算
    Q·K^T / sqrt(d_head) 注意力分数，应用 softmax，并聚合
    值向量。四个投影矩阵（Q, K, V, O）是仅权重量化的
    主要目标。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 四个无偏置的线性投影：查询 Q、键 K、值 V、输出 O
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch（批次）, sequence length（序列长度）, d_model
        # 投影到 Q、K、V，并重塑为多头格式：(B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力：softmax(Q·K^T / sqrt(d)) · V
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = attn_weights @ v  # (B, n_heads, T, head_dim)

        # 合并多头回原始维度：(B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """两层 MLP，使用 GELU 激活函数，典型于 GPT 风格的 Transformer。

    维度：d_model → d_ff → d_model（d_ff 通常是 d_model 的 4 倍）。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    """单个 GPT Transformer 块：pre-norm 注意力 + pre-norm 前馈网络。

    使用 pre-layer-normalization（norm → sublayer → residual add）布局，
    这是现代 GPT 架构中的标准布局。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先层归一化 → 注意力 → 残差加法
        x = x + self.attn(self.ln1(x))
        # 先层归一化 → 前馈网络 → 残差加法
        x = x + self.ffn(self.ln2(x))
        return x


class SmallGPT(nn.Module):
    """一个紧凑的 3 层 GPT 风格自回归语言模型。

    该模型故意设计得很小（约 3-4 百万参数），以便所有量化实验
    可以在 CPU 上舒适运行。架构遵循标准 GPT-2 布局：token + 位置嵌入、
    堆叠的带 pre-norm 的 Transformer 块、最终层归一化，以及
    与 token 嵌入共享权重的输出投影（LM head），以提高参数效率。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Token 嵌入和位置嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # 堆叠的 Transformer 块
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.d_model)

        # LM head：将隐藏状态投影回词汇表 logits。
        # 使用权值共享（与 token_embedding.weight 绑定）。
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # 绑定权重

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """前向传播，返回每个位置的 logits。

        Args:
            idx: 形状为 (B, T) 的 LongTensor，包含 token 索引。

        Returns:
            形状为 (B, T, vocab_size) 的 FloatTensor，包含 logits。
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, (
            f"序列长度 {T} 超过 max_seq_len {self.config.max_seq_len}"
        )

        # Token + 位置嵌入
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)
        pos_emb = self.position_embedding(pos)  # (1, T, d_model)
        x = tok_emb + pos_emb

        # 堆叠的 Transformer 块
        for layer in self.layers:
            x = layer(x)

        # 最终层归一化 + LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def count_parameters(self) -> dict[str, int]:
        """返回按组件分解的参数数量。"""
        counts: dict[str, int] = {}
        for name, param in self.named_parameters():
            counts[name] = param.numel()
        return counts


# =============================================================================
# 第 2 部分：分组 4-bit 量化（AWQ 风格）
# =============================================================================
#
# 激活感知权重量化（AWQ）概述：
#   AWQ 观察到并非所有权重通道对模型输出的贡献相同。
#   与大幅激活（显著通道）相关的通道应该以更高的保真度保留。
#   AWQ 找到每个通道的缩放因子 s_i，以最小化：
#
#       min_s  || W·X - Q(W·diag(s)) · diag(s)^{-1} · X ||
#
#   其中 Q(·) 是量化器。缩放因子 s_i 通过对校准集的激活进行快速
#   网格搜索来找到。
#
#   在本模拟中，我们使用每组（group_size=128）的最小-最大对称量化
#   作为基础，并标注 AWQ 会在何处添加激活感知。真正的 AWQ 还会
#   搜索最优的每通道缩放因子；我们的分组方法已经捕获了局部性优势。
#
# 量化参数：
#   - group_size: 每个独立量化组包含 128 个元素
#   - n_bits:     4 位 → 2^4 = 16 个级别（值 0…15）
#   - scale:      (max - min) / 15，以 FP16 存储
#   - zero:       min 值，以 FP16 存储
#
# 打包方案：
#   两个连续的 4-bit 值被打包到一个 uint8 字节中：
#       packed_byte = (high_nibble << 4) | (low_nibble & 0x0F)
#   这恰好为量化载荷产生每个参数 0.5 字节。

GROUP_SIZE: int = 128
N_BITS: int = 4
Q_MAX: int = (1 << N_BITS) - 1  # 15（即 2^4 - 1）


def _quantize_tensor(
    w: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size, int]:
    """将 float32 张量按组量化到 4-bit 并打包为 uint8。

    Args:
        w:          任意形状的权重张量。
        group_size: 每个量化组中的标量元素数量。

    Returns:
        packed:         打包 4-bit 值的 uint8 张量（长度约 N/2）。
        scales:         每组的缩放因子，float16 张量（长度 = n_groups）。
        zeros:          每组的零点，float16 张量（长度 = n_groups）。
        original_shape: 输入张量的形状，用于重建。
        n_elements:     有效（未填充）元素的数量。
    """
    original_shape = w.shape
    w_flat = w.detach().reshape(-1).float()  # 展平为一维
    n_elements = w_flat.numel()
    n_groups = (n_elements + group_size - 1) // group_size  # 向上取整

    # 填充到 group_size 的倍数，使每个组具有相同数量的元素。
    padded_size = n_groups * group_size
    w_padded = torch.zeros(padded_size, dtype=torch.float32)
    w_padded[:n_elements] = w_flat

    scales = torch.zeros(n_groups, dtype=torch.float16)
    zeros = torch.zeros(n_groups, dtype=torch.float16)
    q_vals = torch.zeros(padded_size, dtype=torch.uint8)

    # 逐组进行最小-最大量化
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = w_padded[start:end]

        w_min = group.min().item()
        w_max = group.max().item()

        if abs(w_max - w_min) < 1e-9:
            # 退化组：所有值相同 — 跳过除法。
            scale = 1.0
            zero = float(w_min)
        else:
            scale = (w_max - w_min) / Q_MAX
            zero = float(w_min)

        scales[g] = scale
        zeros[g] = zero

        # 均匀量化：q = round((w - zero) / scale)，然后限制在 [0, Q_MAX]
        q = torch.round((group - zero) / scale).clamp(0, Q_MAX)
        q_vals[start:end] = q.to(torch.uint8)

    # 将两个 4-bit nibble 打包到一个 uint8 字节中。
    packed_size = padded_size // 2
    packed = torch.zeros(packed_size, dtype=torch.uint8)
    for i in range(packed_size):
        high = q_vals[2 * i].item() & 0x0F  # 高 4 位
        low = q_vals[2 * i + 1].item() & 0x0F  # 低 4 位
        packed[i] = (high << 4) | low

    return packed, scales, zeros, original_shape, n_elements


def _dequantize_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    original_shape: torch.Size,
    n_elements: int,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """将 4-bit 分组量化反向恢复为 float32 张量。

    Args:
        packed:         来自 ``_quantize_tensor`` 的 uint8 打包张量。
        scales:         每组的 float16 缩放因子。
        zeros:          每组的 float16 零点。
        original_shape: 原始权重张量的形状。
        n_elements:     有效元素的数量（填充前）。
        group_size:     必须与量化时使用的值匹配。

    Returns:
        形状为 ``original_shape`` 的去量化 float32 张量。
    """
    packed_size = packed.numel()
    padded_size = packed_size * 2  # 2 × 4-bit 值/字节

    # 解包 nibble。
    q_vals = torch.zeros(padded_size, dtype=torch.float32)
    for i in range(packed_size):
        byte_val = packed[i].item()
        q_vals[2 * i] = float((byte_val >> 4) & 0x0F)  # 高 4 位
        q_vals[2 * i + 1] = float(byte_val & 0x0F)  # 低 4 位

    n_groups = scales.numel()
    w_deq = torch.zeros(padded_size, dtype=torch.float32)

    # 逐组反量化：w = q * scale + zero
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, padded_size)
        q_group = q_vals[start:end]
        w_deq[start:end] = q_group * scales[g].float() + zeros[g].float()

    # 去除填充并重塑回原始形状。
    w_deq = w_deq[:n_elements].reshape(original_shape)
    return w_deq


def quantize_linear_layers(model: nn.Module) -> dict[str, Any]:
    """收集所有 nn.Linear 权重，量化它们，返回量化数据。

    Embedding 和 LayerNorm 参数有意保留在 FP32。

    Returns:
        将参数名映射到
          {'packed', 'scales', 'zeros', 'original_shape', 'n_elements'} 的字典。
    """
    quantized: dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 仅量化线性层的权重（偏置保留在 FP32）
            packed, scales, zeros, shape, n_el = _quantize_tensor(module.weight.data)
            quantized[name] = {
                "packed": packed,
                "scales": scales,
                "zeros": zeros,
                "original_shape": shape,
                "n_elements": n_el,
            }
    return quantized


def apply_dequantized_weights(model: nn.Module) -> None:
    """原地替换：对每个 nn.Linear 权重进行量化 → 去量化。"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            packed, scales, zeros, shape, n_el = _quantize_tensor(module.weight.data)
            w_deq = _dequantize_tensor(packed, scales, zeros, shape, n_el)
            module.weight.data = w_deq


def compute_quantized_model_size(model: nn.Module) -> dict[str, float]:
    """计算模型在 FP32、FP16 和 INT4 下的总存储（字节）。

    分类通过遍历模块树并针对 ``nn.Linear``（可量化）与
    ``nn.Embedding`` / ``nn.LayerNorm``（不可量化）检查 ``isinstance`` 来完成。
    权重共享参数通过 ``id()`` 检测，因此不会被重复计数。当参数在
    可量化模块（nn.Linear）和不可量化模块（例如 token_embedding）之间共享时，
    它被分类为可量化，因为 ``apply_dequantized_weights`` 例程在实践中会量化它。

    INT4 大小包含打包的 4-bit 载荷（每个权重 0.5 B）加上每组的
    scale + zero 元数据（各 2×2 B FP16）。

    Returns:
        {'fp32_bytes': ..., 'fp16_bytes': ..., 'int4_bytes': ...,
         'quantizable_params': ..., 'non_quantized_params': ...,
         'total_groups': ...}
    """
    # -- 第 1 步：收集 nn.Linear 模块所拥有的参数 ID --------
    quantizable_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            for _, param in module.named_parameters(recurse=False):
                quantizable_ids.add(id(param))

    # -- 第 2 步：去重计数 --------------------------
    seen_ids: set[int] = set()  # 用于跟踪已见参数，避免重复计数
    total_params = 0
    total_quantizable_params = 0
    total_groups = 0

    for module in model.modules():
        for _, param in module.named_parameters(recurse=False):
            pid = id(param)
            if pid in seen_ids:
                continue  # 绑定的权重（例如 lm_head ↔ token_embedding）
            seen_ids.add(pid)

            n = param.numel()
            total_params += n

            if pid in quantizable_ids:
                total_quantizable_params += n
                total_groups += (n + GROUP_SIZE - 1) // GROUP_SIZE  # 向上取整

    # FP32 存储：每个参数 4 字节
    fp32_bytes = total_params * 4
    # FP16 存储：每个参数 2 字节
    fp16_bytes = total_params * 2

    # INT4：打包后的权重（0.5 B/参数）+ 每组元数据。
    packed_weight_bytes = math.ceil(total_quantizable_params / 2)
    metadata_bytes = total_groups * 2 * 2  # scale（2 B）+ zero（2 B），各为 FP16
    int4_bytes_quantizable = packed_weight_bytes + metadata_bytes

    # 不可量化的参数保持 FP32（embedding + layernorm）。
    non_quantized_params = total_params - total_quantizable_params
    non_quantized_bytes = non_quantized_params * 4
    int4_bytes = int4_bytes_quantizable + non_quantized_bytes

    return {
        "fp32_bytes": float(fp32_bytes),
        "fp16_bytes": float(fp16_bytes),
        "int4_bytes": float(int4_bytes),
        "quantizable_params": float(total_quantizable_params),
        "non_quantized_params": float(non_quantized_params),
        "total_groups": float(total_groups),
    }


# =============================================================================
# 第 3 部分：困惑度测量
# =============================================================================


@torch.no_grad()
def compute_perplexity(model: nn.Module, input_ids: torch.Tensor) -> float:
    """计算自回归模型在 token 序列上的困惑度。

    困惑度 = exp(cross-entropy loss)，其中 loss 通过使用每个 token
    的前缀来预测该 token（标准因果 LM 设置）来计算。

    Args:
        model:     处于 eval 模式的 SmallGPT 模型。
        input_ids: 形状为 (1, T) 的 LongTensor。

    Returns:
        以 Python float 形式返回的困惑度。
    """
    model.eval()
    logits = model(input_ids)  # (1, T, vocab_size)

    # 移位：使用 token 0..T-2 来预测 token 1..T-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # 交叉熵损失计算
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return math.exp(loss.item())


# =============================================================================
# 第 4 部分：KV 缓存大小计算
# =============================================================================


def compute_kv_cache_size(
    config: GPTConfig,
    seq_len: int,
    kv_dtype_bytes: int,
    batch_size: int = 1,
) -> float:
    """计算给定序列长度的 KV 缓存内存占用。

    KV 缓存为每个 Transformer 层存储一个 Key 和一个 Value 张量。
    每个的形状为 (batch_size, n_heads, seq_len, head_dim)。

    Args:
        config:         GPT 模型配置。
        seq_len:        当前上下文/序列长度。
        kv_dtype_bytes: 每个元素的字节数（FP16 为 2，INT8 为 1）。
        batch_size:     批次大小（单序列推理默认为 1）。

    Returns:
        KV 缓存的总大小（字节）。
    """
    head_dim = config.d_model // config.n_heads
    # 每个张量的元素数
    per_tensor_elements = batch_size * config.n_heads * seq_len * head_dim
    # 每层 2 个张量（K + V），共 config.n_layers 层
    total_elements = 2 * config.n_layers * per_tensor_elements
    return float(total_elements * kv_dtype_bytes)


# =============================================================================
# 第 5 部分：FlashAttention 概念讲解
# =============================================================================
#
# 下面的注释解释了 FlashAttention（Dao et al., 2022）背后的核心思想。
# 不提供实际实现；目的是建立对算法的概念性理解。
#
# ---------------------------------------------------------------------------
# 问题：标准注意力受内存限制
# ---------------------------------------------------------------------------
#
#   S = Q @ K^T          # (N, N) 注意力分数  ← O(N²) 内存！
#   P = softmax(S)        # (N, N) 注意力权重 ← O(N²) 内存！
#   O = P @ V            # (N, d) 输出
#
# S 和 P 都是 N×N 矩阵。对于 N = 2048 个 token 和 FP16，这是
# 2048² × 2 = 8.4 MB/头 × 4 个头 = 33.6 MB。对于 N = 8192 个 token：
# 8192² × 2 = 134 MB/头 — 呈二次方爆炸式增长。在 GPU 上，这些
# 中间张量必须写入和读取自高带宽内存（HBM），其带宽（~1-2 TB/s）是瓶颈。
# 实际的矩阵乘法很快（tensor core 可提供 ~312 TFLOPS）；问题在于
# 我们大部分时间花在数据搬移上，而不是计算上。
#
# ---------------------------------------------------------------------------
# 核心洞察：分块计算注意力
# ---------------------------------------------------------------------------
#
# FlashAttention 将 Q, K, V 划分为可以装入快速片上 SRAM 的块（tile）
# （A100 上每个 SM 约 100 KB）。与其在 HBM 中物化完整的 N×N 注意力矩阵，我们：
#
#   1. 将一块 Q 行加载到 SRAM 中          [大小：Br × d]
#   2. 将一块 K, V 行加载到 SRAM 中       [大小：Bc × d]
#   3. 计算局部注意力分数 S_block = Q_block @ K_block^T  [Br × Bc]
#   4. 使用在线 softmax 重新缩放（见下文）
#   5. 累积输出 O_block += P_block @ V_block，按需重新缩放
#   6. 将最终的 O_block 写回 HBM
#
# 通过为每个 Q 块迭代 K,V 块，我们永远不需要在 HBM 中保存完整的
# N×N 注意力矩阵。HBM 流量从 O(N²) 降至 O(N)。
#
# ---------------------------------------------------------------------------
# 在线 Softmax：无需完整向量即可计算 softmax
# ---------------------------------------------------------------------------
#
# 标准 softmax 需要对分数向量 s 进行两次遍历：
#
#   m = max(s)                    # 第 1 次：找到最大值以保证数值稳定性
#   p = exp(s - m) / sum(exp(s - m))  # 第 2 次：指数化和归一化
#
# 在线 softmax 维护运行时的最大值和求和值：
#
#   def online_softmax_update(s_block, m_old, l_old, O_old, V_block):
#       m_new = max(m_old, max(s_block))                         # 更新运行最大值
#       l_new = exp(m_old - m_new) * l_old + sum(exp(s_block - m_new))  # 更新归一化分母
#       # 重新缩放旧输出并添加新贡献
#       O_new = (l_old / l_new) * exp(m_old - m_new) * O_old
#             + exp(s_block - m_new) / l_new * V_block
#       return m_new, l_new, O_new
#
# 这使我们能够一次处理一个 K,V 块，增量式地累积最终的
# softmax 归一化输出。
#
# ---------------------------------------------------------------------------
# SRAM vs HBM：为什么分块很重要
# ---------------------------------------------------------------------------
#
# 在现代 GPU 上（例如 NVIDIA A100）：
#   - HBM 容量：   40-80 GB，带宽 ~2 TB/s
#   - SRAM 容量：  每个 SM 约 192 KB（总共 20 MB），带宽 ~19 TB/s
#
# 标准注意力将 S 和 P 存储在 HBM 中 → 受限于 HBM 带宽。
# FlashAttention 将所有中间块保留在 SRAM 中 → 受限于
# 计算吞吐量。结果是 2-4 倍的实际时钟速度提升和 10-20 倍
# 的内存节省（在长序列上）。
#
# ---------------------------------------------------------------------------
# FlashAttention 中的因果掩码
# ---------------------------------------------------------------------------
#
# 对于自回归（GPT 风格）模型，注意力掩码是下三角的。
# FlashAttention 通过仅加载对当前 Q 块未被掩蔽的 K,V 块来处理这一点。
# 具体来说，对于覆盖行 i*Br … (i+1)*Br 的 Q 块 i，
# 仅加载列索引 ≤ (i+1)*Br - 1 的 K,V 块。这进一步减少了
# 早期 token 的内存流量。
#
# ---------------------------------------------------------------------------
# 为什么我们在这里*不*实现它
# ---------------------------------------------------------------------------
#
# 正确的 FlashAttention 实现需要编写自定义 CUDA 内核
# （或 Triton 内核），仔细管理共享内存、
# 线程块分块和 warp 级别的 tensor core 指令。
# 本模拟仅在 CPU 上运行，因此我们仅限概念性
# 解释。对生产级实现感兴趣的读者
# 应参考官方 FlashAttention 仓库和 Dao et al.
# （2022）的论文 "FlashAttention: Fast and Memory-Efficient Exact Attention
# with IO-Awareness"。
#
# 关键要点：
#   - FlashAttention 通过分块实现精确注意力，无需 O(N²) 内存
#   - 在线 softmax 允许增量归一化
#   - 结果：2-4 倍加速，10-20 倍内存节省
#   - 需要自定义 IO-aware 内核（CUDA/Triton）


# =============================================================================
# 第 6 部分：主演示
# =============================================================================


def _format_bytes(b: float) -> str:
    """将字节数格式化为人类可读的字符串。"""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024.0:
            return f"{b:,.1f} {unit}"
        b /= 1024.0
    return f"{b:,.1f} TB"


def _separator(title: str) -> None:
    """打印格式化的章节分隔符。"""
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> None:
    """运行完整的 LLM 部署模拟。

    步骤：
      1. 实例化小型 GPT 模型。
      2. 在合成文本上测量 FP32 困惑度。
      3. 将权重量化到 4-bit（分组），测量 INT4 困惑度。
      4. 比较模型大小（FP32 / FP16 / INT4）。
      5. 在不同上下文长度下比较 KV 缓存大小（FP16 vs INT8）。
      6. 打印 FlashAttention 概念解释。
    """

    # ------------------------------------------------------------------
    # 设置随机种子以确保可重复性
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. 构建模型
    # ------------------------------------------------------------------
    config = GPTConfig()
    model = SmallGPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"模型已构建：{total_params:,} 个总参数 "
        f"（{config.n_layers} 层，d_model={config.d_model}，"
        f"n_heads={config.n_heads}）"
    )

    # 打印详细的参数分解
    print("\n参数分解：")
    counts = model.count_parameters()
    total = sum(counts.values())
    for name, cnt in sorted(counts.items()):
        pct = 100.0 * cnt / total
        print(f"  {name:<55s} {cnt:>10,}  ({pct:5.1f}%)")
    print(f"  {'总计':<55s} {total:>10,}")

    # ------------------------------------------------------------------
    # 2. 合成文本生成 & 基线困惑度
    # ------------------------------------------------------------------
    _separator("困惑度测量（FP32 基线）")

    # 创建一个合成序列：来自我们词汇表的 64 个随机 token。
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    print(
        f"合成输入序列：shape={tuple(input_ids.shape)}，vocab_size={config.vocab_size}"
    )

    ppl_fp32 = compute_perplexity(model, input_ids)
    print(f"FP32 困惑度：{ppl_fp32:.4f}")

    # ------------------------------------------------------------------
    # 3. 仅权重量化 → INT4 困惑度
    # ------------------------------------------------------------------
    _separator("仅权重量化（分组 4-bit）")

    # 深拷贝模型以保留原始 FP32 权重。
    model_int4 = copy.deepcopy(model)

    # 原地对所有 nn.Linear 权重进行量化-再-去量化。
    apply_dequantized_weights(model_int4)

    ppl_int4 = compute_perplexity(model_int4, input_ids)
    delta_ppl = ppl_int4 - ppl_fp32
    print(f"INT4 困惑度：{ppl_int4:.4f}")
    print(f"困惑度退化：{delta_ppl:+.4f} （{100.0 * delta_ppl / ppl_fp32:+.2f}%）")
    print(
        "\n注意：由于模型具有随机初始化（未训练）的权重，并且是在随机合成 token 上"
        "进行评估，\n因此绝对困惑度值极高。\n在适当训练的模型上，困惑度约为 ~10-50，"
        "使用分组量化时 INT4 退化\n通常保持在 1-5% 以下。相对变化\n仍然说明了"
        "量化噪声如何影响输出质量。"
    )

    # ------------------------------------------------------------------
    # 4. 模型大小比较
    # ------------------------------------------------------------------
    _separator("模型大小比较")

    sizes = compute_quantized_model_size(model)
    fp32_b = sizes["fp32_bytes"]
    fp16_b = sizes["fp16_bytes"]
    int4_b = sizes["int4_bytes"]

    print(f"{'格式':<8s} {'大小':>14s} {'对比 FP32 的比例':>18s}")
    print("-" * 40)
    print(f"{'FP32':<8s} {_format_bytes(fp32_b):>14s} {'1.00x（基线）':>18s}")
    print(f"{'FP16':<8s} {_format_bytes(fp16_b):>14s} {f'{fp32_b / fp16_b:.2f}x':>18s}")
    print(f"{'INT4':<8s} {_format_bytes(int4_b):>14s} {f'{fp32_b / int4_b:.2f}x':>18s}")
    print()
    print(
        f"可量化参数：{sizes['quantizable_params']:,.0f} "
        f"（占总数的 {100.0 * sizes['quantizable_params'] / total_params:.1f}%）"
    )
    print(f"不可量化参数（嵌入 + 归一化层）：{sizes['non_quantized_params']:,.0f}")
    print(f"量化组数（group_size={GROUP_SIZE}）：{sizes['total_groups']:,.0f}")

    # 详细的 INT4 分解
    packed_bytes = math.ceil(sizes["quantizable_params"] / 2)
    metadata_bytes = int4_b - sizes["non_quantized_params"] * 4 - packed_bytes
    print(f"\nINT4 存储分解：")
    print(f"  打包的 4-bit 权重：      {_format_bytes(packed_bytes):>14s}")
    print(
        f"  scale + zero 元数据：    {_format_bytes(metadata_bytes):>14s} "
        f"（FP16，每组 2×2 B）"
    )
    print(
        f"  不可量化（FP32）：        "
        f"{_format_bytes(sizes['non_quantized_params'] * 4):>14s}"
    )

    # ------------------------------------------------------------------
    # 5. KV 缓存大小比较
    # ------------------------------------------------------------------
    _separator("KV 缓存大小：FP16 vs INT8")

    context_lengths = [256, 512, 1024, 2048]  # 模拟的上下文长度
    print(
        f"{'上下文':>8s}  {'FP16 KV 缓存':>16s}  {'INT8 KV 缓存':>16s}  {'缩减':>12s}"
    )
    print("-" * 64)

    for L in context_lengths:
        # FP16 每个元素 2 字节，INT8 每个元素 1 字节
        kv_fp16 = compute_kv_cache_size(config, L, kv_dtype_bytes=2)
        kv_int8 = compute_kv_cache_size(config, L, kv_dtype_bytes=1)
        reduction = 100.0 * (1.0 - kv_int8 / kv_fp16) if kv_fp16 > 0 else 0.0

        print(
            f"{L:>8d}  "
            f"{_format_bytes(kv_fp16):>16s}  "
            f"{_format_bytes(kv_int8):>16s}  "
            f"{reduction:>9.1f}%"
        )

    # 显示最长上下文下每层的分解。
    L_max = context_lengths[-1]
    head_dim = config.d_model // config.n_heads
    per_layer_fp16 = 2 * config.n_heads * L_max * head_dim * 2  # 每层 K + V
    print(f"\n在 L={L_max} 时每层 KV 缓存：")
    print(
        f"  K 张量：(1, {config.n_heads}, {L_max}, {head_dim}) "
        f"= {_format_bytes(float(config.n_heads * L_max * head_dim * 2))} FP16"
    )
    print(
        f"  V 张量：(1, {config.n_heads}, {L_max}, {head_dim}) "
        f"= {_format_bytes(float(config.n_heads * L_max * head_dim * 2))} FP16"
    )
    print(f"  每层（K+V）：{_format_bytes(float(per_layer_fp16))} FP16")
    print(
        f"  总计（{config.n_layers} 层）："
        f"{_format_bytes(float(per_layer_fp16 * config.n_layers))} FP16"
    )

    # ------------------------------------------------------------------
    # 6. 总结 & FlashAttention 说明
    # ------------------------------------------------------------------
    _separator("总结")

    print("权重量化：")
    print(f"  - 分组 4-bit（group_size={GROUP_SIZE}）")
    print(
        f"  - 困惑度：{ppl_fp32:.4f} (FP32) → {ppl_int4:.4f} (INT4) "
        f"（{delta_ppl:+.4f}）"
    )
    print(
        f"  - 模型大小：{_format_bytes(fp32_b)} → {_format_bytes(int4_b)} "
        f"（缩小了 {fp32_b / int4_b:.1f} 倍）"
    )

    print("\nKV 缓存量化：")
    for L in context_lengths:
        kv_fp16 = compute_kv_cache_size(config, L, kv_dtype_bytes=2)
        kv_int8 = compute_kv_cache_size(config, L, kv_dtype_bytes=1)
        print(
            f"  - L={L:>4d}：{_format_bytes(kv_fp16):>10s} (FP16) → "
            f"{_format_bytes(kv_int8):>10s} (INT8) "
            f"（缩减了 {100.0 * (kv_fp16 - kv_int8) / kv_fp16:.0f}%）"
        )

    print("\nFlashAttention：")
    print("  请参见上方第 5 部分中广泛的内联注释，了解分块算法、")
    print("  在线 softmax 以及 SRAM vs HBM 内存层次的完整概念讲解。")
    print("  关键要点：HBM 流量从 O(N²) 降至 O(N)，")
    print("  实现 2-4 倍加速和 10-20 倍内存节省。")

    print("\n" + "=" * 72)
    print("  模拟完成。所有计算均在 CPU 上执行。")
    print("=" * 72)


if __name__ == "__main__":
    main()
