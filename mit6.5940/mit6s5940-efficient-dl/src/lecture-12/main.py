"""
Transformer 效率分析 (第12讲)
==============================================
构建一个玩具级 GPT 风格的 Transformer, 并进行分析性对比:
  - MHA (多头注意力 Multi-Head Attention)
  - MQA (多查询注意力 Multi-Query Attention)
  - GQA (分组查询注意力 Grouped-Query Attention)

执行的分析:
  1. 不同序列长度下前向传播的 FLOPs 计数 (注意力的二次增长)
  2. 不同序列长度下 KV-cache 的内存占用 (FP16)
  3. 多种模型尺寸下 MHA / MQA / GQA 的参数数量对比

所有计算仅在 CPU 上进行。无需 CUDA。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 配置类
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    """GPT 风格 Transformer 的配置参数。"""

    n_layers: int = 12  # Transformer 块的数量
    d_model: int = 768  # 隐藏维度 (模型宽度)
    n_heads: int = 12  # 查询 (Query) 头数
    n_kv_heads: int = 12  # 键/值 (Key/Value) 头数 (12=MHA, 1=MQA, 4=GQA)
    head_dim: int = 64  # 每个头的维度 (d_model = n_heads * head_dim)
    vocab_size: int = 50257  # 词汇表大小 (GPT-2 默认值)
    max_seq_len: int = 1024  # 最大序列长度
    ffn_multiplier: int = 4  # FFN 隐藏维度 = d_model * ffn_multiplier

    def __post_init__(self) -> None:
        """初始化后验证配置的一致性。

        1. d_model 必须等于 n_heads * head_dim
        2. n_heads 必须能被 n_kv_heads 整除 (GQA 的分组约束)
        """
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) 必须等于 n_heads ({self.n_heads}) "
            f"* head_dim ({self.head_dim})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) 必须能被 n_kv_heads ({self.n_kv_heads}) 整除"
        )

    @property
    def n_groups(self) -> int:
        """共享一个 KV 头的查询头组数 (GQA 特有)。"""
        return self.n_heads // self.n_kv_heads


# ---------------------------------------------------------------------------
# 注意力模块
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """标准多头注意力 (MHA)。

    每个查询头都有独立的键和值投影。
    n_kv_heads == n_heads。
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_heads  # MHA 中 KV 头数 = Q 头数
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model

        # Q, K, V, O 投影矩阵 (无偏置, 遵循 GPT 惯例)
        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MHA 前向传播。

        Q, K, V 形状均为 (B, n_heads, S, head_dim)。
        使用因果掩码 (causal) 以确保自回归属性。
        """
        B, S, D = x.shape
        # 投影 + 拆分为多头: (B, S, D) → (B, n_heads, S, head_dim)
        q = self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力 (带因果掩码)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # 合并多头输出: (B, n_heads, S, head_dim) → (B, S, D)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


class MultiQueryAttention(nn.Module):
    """多查询注意力 (MQA)。

    所有查询头共享同一个键/值投影。
    n_kv_heads == 1。

    MQA 的优势: KV 投影矩阵大小减少为原来的 1/n_heads,
    且 KV-cache 内存同样减少为原来的 1/n_heads。
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = 1  # MQA 中只有一个 KV 头
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model

        # Q 投影大小不变; K 和 V 投影仅输出 head_dim 维度 (而非 n_heads * head_dim)
        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.W_k = nn.Linear(cfg.d_model, 1 * cfg.head_dim, bias=False)
        self.W_v = nn.Linear(cfg.d_model, 1 * cfg.head_dim, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MQA 前向传播。

        K 和 V 被广播 (expand) 到所有查询头, 然后进行标准的缩放点积注意力。
        """
        B, S, D = x.shape
        # Q: (B, n_heads, S, head_dim)
        q = self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # K, V: (B, 1, S, head_dim) — 只有一个 KV 头
        k = self.W_k(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 将 K/V 广播到所有查询头: (B, 1, S, head_dim) → (B, n_heads, S, head_dim)
        k = k.expand(B, self.n_heads, S, self.head_dim)
        v = v.expand(B, self.n_heads, S, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


class GroupedQueryAttention(nn.Module):
    """分组查询注意力 (GQA)。

    查询头被划分为若干组; 每组共享一个 KV 头。
    1 < n_kv_heads < n_heads。

    GQA 在 MHA (最高质量) 和 MQA (最高效率) 之间取得平衡。
    例如 n_heads=12, n_kv_heads=4 意味着每 3 个查询头共享一个 KV 头。
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        self.n_groups = cfg.n_groups  # n_heads // n_kv_heads

        # Q 投影: 所有头独立
        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        # K, V 投影: 仅 n_kv_heads 个头
        self.W_k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.W_o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GQA 前向传播。

        每个 KV 头通过 unsqueeze + expand 复制 n_groups 次,
        使得每个查询头都有一个对应的 K/V 头参与计算。
        """
        B, S, D = x.shape
        # Q: (B, n_heads, S, head_dim)
        q = self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # K, V: (B, n_kv_heads, S, head_dim)
        k = self.W_k(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 将每个 KV 头重复 n_groups 次, 使每个 Q 头都有对应的 K/V 头
        # expand: (B, n_kv, 1, S, hd) → (B, n_kv, n_groups, S, hd)
        k = k.unsqueeze(2).expand(B, self.n_kv_heads, self.n_groups, S, self.head_dim)
        # reshape: → (B, n_heads, S, head_dim)
        k = k.reshape(B, self.n_heads, S, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.n_kv_heads, self.n_groups, S, self.head_dim)
        v = v.reshape(B, self.n_heads, S, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Transformer 块
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """单个 Transformer 块: 注意力 + FFN, 使用 Pre-Norm 残差连接。

    Pre-Norm 架构 (先 LayerNorm 再注意力/FFN) 是 GPT 系列的标准做法,
    相比 Post-Norm 训练更稳定。
    """

    def __init__(self, attention: nn.Module, cfg: GPTConfig) -> None:
        super().__init__()
        self.attn = attention  # 注意力子层 (MHA / MQA / GQA)
        self.ln1 = nn.LayerNorm(cfg.d_model)  # 注意力前的归一化
        self.ln2 = nn.LayerNorm(cfg.d_model)  # FFN 前的归一化
        # FFN 子层: 两层线性 + GELU 激活 (SwiGLU 的简化版)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_multiplier * cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ffn_multiplier * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-Norm 残差前向传播。

        x = x + Attn(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# GPT 模型
# ---------------------------------------------------------------------------


class GPT(nn.Module):
    """玩具级 GPT 风格 Transformer 模型。

    通过 config.n_kv_heads 支持 MHA、MQA 或 GQA。
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token 嵌入 + 位置嵌入 (绝对位置编码)
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # 根据 n_kv_heads 选择注意力变体
        if cfg.n_kv_heads == cfg.n_heads:
            attn_cls = MultiHeadAttention  # MHA
        elif cfg.n_kv_heads == 1:
            attn_cls = MultiQueryAttention  # MQA
        else:
            attn_cls = GroupedQueryAttention  # GQA

        # 堆叠 n_layers 个 Transformer 块
        self.blocks = nn.Sequential(
            *[TransformerBlock(attn_cls(cfg), cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = nn.LayerNorm(cfg.d_model)  # 最终 LayerNorm
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # 语言模型头

        # 权重绑定 (GPT 模型的标准做法): LM head 权重与 token embedding 共享
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """GPT 前向传播。

        Parameters
        ----------
        token_ids : (B, S)
            输入的 token ID 序列。

        Returns
        -------
        logits : (B, S, vocab_size)
            每个位置的 logits 输出。
        """
        B, S = token_ids.shape
        # 生成位置索引: (1, S)
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0)
        # Token 嵌入 + 位置嵌入
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        # 通过所有 Transformer 块
        x = self.blocks(x)
        # 最终 LayerNorm + LM head 投影
        x = self.ln_final(x)
        return self.lm_head(x)  # (B, S, vocab_size)


# ---------------------------------------------------------------------------
# FLOPs 分析
# ---------------------------------------------------------------------------


def flops_matmul(m: int, n: int, k: int) -> int:
    """计算矩阵乘法 (m × k) @ (k × n) → (m × n) 的 FLOPs。

    每个输出元素需要 k 次乘加运算 = 2*k 次浮点操作。
    总计: 2 * m * n * k FLOPs。

    这是标准的 "2*" 约定, 每次 MAC 计为 2 FLOPs。
    """
    return 2 * m * n * k


def compute_flops(
    seq_len: int,
    n_layers: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    ffn_multiplier: int = 4,
) -> Tuple[int, int]:
    """计算 Transformer 一次前向传播的分析性 FLOPs。

    Returns
    -------
    (attention_flops, total_flops) : (int, int)
        attention_flops: 仅注意力部分的 FLOPs (不含线性投影)
        total_flops: 总 FLOPs (包含所有矩阵乘法和 LM head)

    FLOPs 按 2* 约定计数: 每次乘加 = 2 FLOPs。
    仅计算矩阵乘法操作 (线性层 + 注意力计算)。
    LayerNorm、GELU、softmax 和嵌入查找忽略不计, 以保持清晰。

    注意力 FLOPs 呈 O(seq_len²) 增长:
        Q @ K^T   →  2 * n_layers * n_heads * seq_len² * head_dim
        Attn @ V  →  2 * n_layers * n_heads * seq_len² * head_dim
        总计      →  4 * n_layers * d_model * seq_len²
    """
    S = seq_len
    d_kv = n_kv_heads * head_dim  # K/V 投影的总维度

    # --- 每层 FLOPs ---

    # Q 投影: (S, d_model) × (d_model, d_model)
    flops_q = flops_matmul(S, d_model, d_model)
    # K 投影: (S, d_model) × (d_model, d_kv)
    flops_k = flops_matmul(S, d_kv, d_model)
    # V 投影: 与 K 相同
    flops_v = flops_k
    # Q @ K^T: 每个 Q 头对每个 K 头的点积运算
    #   per head: (S, head_dim) × (head_dim, S) with KV broadcasting
    flops_qkt = flops_matmul(S, S, head_dim) * n_heads
    # Attn @ V: 每个头: (S, S) @ (S, head_dim)
    flops_attnv = flops_matmul(S, head_dim, S) * n_heads
    # O 投影: (S, d_model) @ (d_model, d_model)
    flops_out = flops_matmul(S, d_model, d_model)
    # FFN 上投影: (S, d_model) @ (d_model, ffn*d_model)
    flops_ffn_up = flops_matmul(S, ffn_multiplier * d_model, d_model)
    # FFN 下投影: (S, ffn*d_model) @ (ffn*d_model, d_model)
    flops_ffn_down = flops_matmul(S, d_model, ffn_multiplier * d_model)

    # 每层 FLOPs 分类汇总
    flops_per_layer_qkv = flops_q + flops_k + flops_v  # 线性投影
    flops_per_layer_attn = flops_qkt + flops_attnv  # O(S²) 注意力计算项
    flops_per_layer_linear = flops_out + flops_ffn_up + flops_ffn_down  # 其他线性层

    # --- 总计 ---
    attention_flops = n_layers * flops_per_layer_attn
    total_flops = (
        n_layers * (flops_per_layer_qkv + flops_per_layer_attn + flops_per_layer_linear)
        + flops_matmul(S, vocab_size, d_model)  # LM head 投影
    )

    return attention_flops, total_flops


# ---------------------------------------------------------------------------
# KV-Cache 内存分析
# ---------------------------------------------------------------------------


def compute_kv_cache_bytes(
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bytes_per_element: int = 2,  # FP16 精度, 每个元素 2 字节
) -> int:
    """计算给定序列长度下的 KV-cache 内存占用 (字节)。

    每层存储:
      - Key cache:   seq_len * n_kv_heads * head_dim 个元素
      - Value cache: seq_len * n_kv_heads * head_dim 个元素

    关键洞察: MQA / GQA 大幅减少 KV-cache 大小,
    因为 n_kv_heads 远小于 MHA 的 n_heads。

    Returns
    -------
    kv_cache_bytes : int
        KV-cache 总内存占用 (字节)。
    """
    elements_per_layer = 2 * seq_len * n_kv_heads * head_dim  # K + V 两个缓存
    return n_layers * elements_per_layer * bytes_per_element


# ---------------------------------------------------------------------------
# 参数数量分析
# ---------------------------------------------------------------------------


def count_attention_params(
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> int:
    """计算注意力子层的参数数量 (无偏置)。

    仅计算 Q, K, V, O 权重矩阵。

    MHA: params = d_model² * (3 + 1) = 4 * d_model²
    MQA: params = d_model² (Q) + 2*d_model*head_dim (K,V) + d_model² (O)
         = 2*d_model² + 2*d_model*head_dim
    """
    d_kv = n_kv_heads * head_dim
    params_q = d_model * d_model  # Q 投影
    params_k = d_model * d_kv  # K 投影
    params_v = d_model * d_kv  # V 投影
    params_o = d_model * d_model  # O 投影
    return params_q + params_k + params_v + params_o


def count_total_params(
    n_layers: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    max_seq_len: int = 1024,
    ffn_multiplier: int = 4,
) -> int:
    """计算模型的总参数数量。

    与 GPT 模型实现匹配:
    - 存在位置嵌入
    - LM head 与 token embedding 权重绑定 (仅计一次)
    - 偏置仅存在于 LayerNorm 层中
    """
    # Token 嵌入 (因权重绑定, 与 LM head 共享)
    emb = vocab_size * d_model
    # 位置嵌入
    pos_emb = max_seq_len * d_model
    # 每块的参数
    attn = count_attention_params(d_model, n_heads, n_kv_heads, head_dim)
    ffn = 2 * d_model * ffn_multiplier * d_model  # FFN: W1 + W2, 无偏置
    ln = 2 * d_model * 2  # 每块两个 LayerNorm (权重 + 偏置, 各 d_model 个)
    per_layer = attn + ffn + ln
    # 最终 LayerNorm
    final_ln = 2 * d_model
    return emb + pos_emb + n_layers * per_layer + final_ln


# ---------------------------------------------------------------------------
# 打印辅助函数
# ---------------------------------------------------------------------------


def _human_readable(num: int) -> str:
    """将数字格式化为人类可读形式 (K, M, B, T)。

    例如: 1234567 → "1.23M"
    """
    if abs(num) >= 1e12:
        return f"{num / 1e12:.2f}T"
    elif abs(num) >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def print_header(title: str) -> None:
    """打印居中的章节标题分隔线。"""
    width = 78
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_flops_table(
    seq_lengths: List[int],
    n_layers: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    vocab_size: int,
) -> None:
    """打印展示 FLOPs 分布与序列长度关系的表格。

    输出包含:
      - 各序列长度下的注意力 FLOPs 和总 FLOPs
      - 注意力 FLOPs 占比 (%)
      - KV-cache 内存占用 (MB)
      - 二次增长验证 (实际比值 vs 预期 (S/S0)²)
    """
    header = (
        f"{'Seq Len':>8s}  {'Attn FLOPs':>14s}  {'Total FLOPs':>14s}  "
        f"{'Attn %':>8s}  {'KV Cache (MB)':>14s}"
    )
    sep = "-" * len(header)

    print()
    print(
        f"  模型: n_layers={n_layers}, d_model={d_model}, "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}"
    )
    print(f"  (Q @ K^T + Attn @ V 按 O(S²) 增长, 线性投影按 O(S) 增长)")
    print()
    print(sep)
    print(header)
    print(sep)

    # 逐序列长度输出统计
    for s in seq_lengths:
        attn_flops, total_flops = compute_flops(
            s,
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
        )
        kv_bytes = compute_kv_cache_bytes(s, n_layers, n_kv_heads, head_dim)
        kv_mb = kv_bytes / (1024 * 1024)  # 转换为 MB
        attn_pct = attn_flops / total_flops * 100  # 注意力占比
        print(
            f"{s:>8d}  "
            f"{_human_readable(attn_flops):>14s}  "
            f"{_human_readable(total_flops):>14s}  "
            f"{attn_pct:>7.1f}%  "
            f"{kv_mb:>14.2f}"
        )

    print(sep)
    print()

    # 显式展示二次增长比率验证
    base_s = seq_lengths[0]
    base_attn, _ = compute_flops(
        base_s,
        n_layers,
        d_model,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
    )
    print("  二次增长验证 (注意力 FLOPs vs 序列长度):")
    print(
        f"  {'Seq Len':>8s}  {'Attn FLOPs':>14s}  {'Ratio':>8s}  {'预期 (S/S0)²':>18s}"
    )
    print(f"  {'-' * 8}  {'-' * 14}  {'-' * 8}  {'-' * 18}")
    for s in seq_lengths:
        attn_flops, _ = compute_flops(
            s,
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
        )
        ratio = attn_flops / base_attn  # 实际增长比
        expected = (s / base_s) ** 2  # 理论二次增长比
        print(
            f"  {s:>8d}  "
            f"{_human_readable(attn_flops):>14s}  "
            f"{ratio:>7.1f}x  "
            f"{expected:>18.1f}"
        )


def print_comparison_table() -> None:
    """打印 MHA vs MQA vs GQA 的参数数量对比表。

    对比多种模型尺寸 (Small / Medium / Large / XL),
    展示不同注意力变体对参数量的影响。
    """
    configs = [
        # (标签, n_layers, d_model, n_heads, n_kv_heads, head_dim, vocab_size)
        ("Small  (d=512, h=8)", 12, 512, 8, 8, 64, 50257),
        ("Medium (d=768, h=12)", 12, 768, 12, 12, 64, 50257),
        ("Large  (d=1024,h=16)", 12, 1024, 16, 16, 64, 50257),
        ("XL     (d=2048,h=32)", 12, 2048, 32, 32, 64, 50257),
    ]

    header = (
        f"{'配置':>22s}  {'MHA 参数':>14s}  {'MQA 参数':>14s}  "
        f"{'GQA-4 参数':>14s}  {'GQA-2 参数':>14s}"
    )
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for label, nl, dm, nh, _, hd, vs in configs:
        # 计算各变体的参数量
        # MHA:  n_kv_heads = n_heads
        # MQA:  n_kv_heads = 1
        # GQA-4: n_kv_heads = 4 (4 组)
        # GQA-2: n_kv_heads = 2 (2 组)
        params_mha = count_total_params(nl, dm, nh, nh, hd, vs)
        params_mqa = count_total_params(nl, dm, nh, 1, hd, vs)
        params_gqa4 = count_total_params(nl, dm, nh, 4, hd, vs) if nh >= 4 else None
        params_gqa2 = count_total_params(nl, dm, nh, 2, hd, vs) if nh >= 2 else None

        gqa4_str = _human_readable(params_gqa4) if params_gqa4 else "N/A"
        gqa2_str = _human_readable(params_gqa2) if params_gqa2 else "N/A"

        print(
            f"{label:>22s}  "
            f"{_human_readable(params_mha):>14s}  "
            f"{_human_readable(params_mqa):>14s}  "
            f"{gqa4_str:>14s}  "
            f"{gqa2_str:>14s}"
        )

    print(sep)
    print()
    print("  图例:")
    print("    MHA   = 多头注意力   (Multi-Head Attention,   n_kv_heads == n_heads)")
    print("    MQA   = 多查询注意力  (Multi-Query Attention,  n_kv_heads == 1)")
    print("    GQA-4 = 分组查询注意力 (Grouped-Query Attention, n_kv_heads == 4)")
    print("    GQA-2 = 分组查询注意力 (Grouped-Query Attention, n_kv_heads == 2)")
    print("  参数节省来自于更小的 K 和 V 投影矩阵。")


def print_kv_cache_comparison_table() -> None:
    """打印固定模型下 MHA / MQA / GQA 的 KV-cache 内存对比表。

    展示: MQA 将 KV-cache 减少 n_heads 倍,
    GQA 在内存节省和注意力质量之间取得平衡。
    """
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    n_layers = 12
    n_heads = 12
    head_dim = 64

    header = (
        f"{'Seq Len':>8s}  "
        f"{'MHA (MB)':>12s}  "
        f"{'GQA-4 (MB)':>12s}  "
        f"{'GQA-2 (MB)':>12s}  "
        f"{'MQA (MB)':>12s}  "
        f"{'MQA vs MHA':>12s}"
    )
    sep = "-" * len(header)

    print()
    print(
        f"  KV-Cache 内存 (FP16)  -  n_layers={n_layers}, "
        f"d_model={n_heads * head_dim}, n_heads={n_heads}, head_dim={head_dim}"
    )
    print()
    print(sep)
    print(header)
    print(sep)

    for s in seq_lengths:
        # 计算各变体的 KV-cache 内存占用 (MB)
        mha_mb = compute_kv_cache_bytes(s, n_layers, n_heads, head_dim) / (1024 * 1024)
        gqa4_mb = compute_kv_cache_bytes(s, n_layers, 4, head_dim) / (1024 * 1024)
        gqa2_mb = compute_kv_cache_bytes(s, n_layers, 2, head_dim) / (1024 * 1024)
        mqa_mb = compute_kv_cache_bytes(s, n_layers, 1, head_dim) / (1024 * 1024)
        reduction = (1 - mqa_mb / mha_mb) * 100  # MQA 相对于 MHA 的节省百分比

        print(
            f"{s:>8d}  "
            f"{mha_mb:>12.2f}  "
            f"{gqa4_mb:>12.2f}  "
            f"{gqa2_mb:>12.2f}  "
            f"{mqa_mb:>12.2f}  "
            f"{reduction:>10.1f}%"
        )

    print(sep)
    print()
    print("  MQA 将 KV-cache 减少 n_heads 倍 (本例中为 12 倍)。")
    print("  GQA 在内存节省和注意力质量之间进行权衡。")


# ---------------------------------------------------------------------------
# 正确性检查 (快速冒烟测试)
# ---------------------------------------------------------------------------


def run_smoke_test() -> None:
    """验证模型能完成前向传播, 并且 FLOPs 框架产生自洽的结果。

    测试内容:
      1. 三种注意力变体 (MHA / GQA / MQA) 的前向传播
      2. 参数计数与实际模型参数匹配
      3. FLOPs 二次增长验证 (seq_len 64→128 应为 4x)
    """
    print_header("冒烟测试: 前向传播与基本检查")

    cfg = GPTConfig(
        n_layers=2,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        vocab_size=1000,
        max_seq_len=64,
    )

    # 为三种注意力变体分别创建配置
    cfg_mha = GPTConfig(**{**cfg.__dict__, "n_kv_heads": 4})  # MHA
    cfg_gqa = GPTConfig(**{**cfg.__dict__, "n_kv_heads": 2})  # GQA
    cfg_mqa = GPTConfig(**{**cfg.__dict__, "n_kv_heads": 1})  # MQA

    for name, c in [("MHA", cfg_mha), ("GQA", cfg_gqa), ("MQA", cfg_mqa)]:
        model = GPT(c)
        model.eval()
        # 随机 token 输入: (1, 32)
        tokens = torch.randint(0, c.vocab_size, (1, 32))
        with torch.no_grad():
            logits = model(tokens)
        # 验证输出形状
        assert logits.shape == (1, 32, c.vocab_size), f"{name}: 输出形状不正确"
        # 验证参数计数一致性
        total = sum(p.numel() for p in model.parameters())
        expected = count_total_params(
            c.n_layers,
            c.d_model,
            c.n_heads,
            c.n_kv_heads,
            c.head_dim,
            c.vocab_size,
            max_seq_len=c.max_seq_len,
        )
        assert total == expected, f"{name}: 参数计数不匹配: {total} vs {expected}"
        print(
            f"  ✓ {name} 通过  (参数={_human_readable(total)}, "
            f"输出形状={list(logits.shape)})"
        )

    # 验证二次增长在小型配置中的正确性
    attn_64, _ = compute_flops(64, 2, 128, 4, 4, 32, 1000)
    attn_128, _ = compute_flops(128, 2, 128, 4, 4, 32, 1000)
    ratio = attn_128 / attn_64
    print(f"\n  ✓ 注意力 FLOPs 比值 64→128: {ratio:.1f}x  (预期 4.0x)")
    assert abs(ratio - 4.0) < 0.1, f"二次增长关系被破坏: 得到 {ratio}"
    print("  所有冒烟测试通过。\n")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """运行完整的 Transformer 效率分析。

    分析内容:
      1. 冒烟测试 (正确性验证)
      2. FLOPs 与序列长度的关系 (二次增长)
      3. MHA vs MQA vs GQA 参数数量对比
      4. KV-cache 内存占用对比
      5. 关键洞察摘要
    """

    print_header("Transformer 效率分析 (第12讲)")
    print("  演示注意力的二次复杂度 (O(S²)),")
    print("  MHA/MQA/GQA 对参数量的影响, 以及")
    print("  MQA/GQA 带来的 KV-cache 内存节省。")
    print()

    # ------------------------------------------------------------------
    # 1. 冒烟测试
    # ------------------------------------------------------------------
    run_smoke_test()

    # ------------------------------------------------------------------
    # 2. 不同序列长度下的 FLOPs 分析
    # ------------------------------------------------------------------
    print_header("1. FLOPs 与序列长度的关系  (GPT-2 Small 近似配置)")

    seq_lengths = [64, 128, 256, 512, 1024]

    # 默认配置 (MHA)
    print_flops_table(
        seq_lengths=seq_lengths,
        n_layers=12,
        d_model=768,
        n_heads=12,
        n_kv_heads=12,  # MHA: KV 头数 = 查询头数
        head_dim=64,
        vocab_size=50257,
    )

    # MQA/GQA 的注意力 FLOPs 与 MHA 几乎相同
    print("  注意: MQA/GQA 的注意力 FLOPs 与 MHA 相同, 因为每个")
    print("        查询头仍然对所有 S 个 key token 计算注意力。")
    print("        节省体现在 K/V 投影 FLOPs (更小的矩阵) 和")
    print("        KV-cache 内存上, 而非 O(S²) 的注意力计算本身。")

    # ------------------------------------------------------------------
    # 3. MHA vs MQA vs GQA 参数对比
    # ------------------------------------------------------------------
    print_header("2. 参数数量: MHA vs MQA vs GQA")
    print_comparison_table()

    # ------------------------------------------------------------------
    # 4. KV-cache 内存对比
    # ------------------------------------------------------------------
    print_header("3. KV-Cache 内存: MHA vs MQA vs GQA")
    print_kv_cache_comparison_table()

    # ------------------------------------------------------------------
    # 5. 摘要
    # ------------------------------------------------------------------
    print_header("关键洞察摘要")

    print("""
  a) 注意力 FLOPs 与序列长度呈 O(S²) 关系。
     - S 翻倍时, Q@K^T 和 Attn@V 的成本变为四倍。
     - 对于长序列 (S > 512), 注意力主导总 FLOPs。

  b) MQA 和 GQA 不会减少注意力 FLOPs。
     - O(S²) 的 Q@K^T + Attn@V 工作量与 MHA 完全相同。
     - 节省体现在 K/V 线性投影和 KV-cache 内存上。

  c) KV-cache 内存随 S 线性增长, 但被 MQA/GQA 大幅削减:
     - MQA:  n_kv_heads = 1    → MHA 缓存大小的 1/n_heads
     - GQA:  n_kv_heads = g    → MHA 缓存大小的 g/n_heads

  d) MQA/GQA 带来的参数减少真实存在但幅度有限:
     - 注意力参数量从 4*d_model² 缩减到约 2*d_model²。
     - FFN 和嵌入参数 (模型的大头) 不发生变化。
     - 采用 MQA/GQA 的主要动机是 KV-cache 内存, 而非 FLOPs。
""")


if __name__ == "__main__":
    main()
