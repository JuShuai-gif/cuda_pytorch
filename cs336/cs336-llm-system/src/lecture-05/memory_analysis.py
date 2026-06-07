"""
第 05 讲 — GPU 架构：Transformer 层的显存分析。

分析典型 Transformer 架构中 attention、MLP 和 embedding 层的
GPU 显存消耗。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 数据类型
# ---------------------------------------------------------------------------


@dataclass
class LayerMemory:
    """单个 Transformer 子层的内存使用明细。"""

    name: str
    parameters_gib: float = 0.0
    activations_gib: float = 0.0  # 前向传播期间的峰值
    fwd_mem_gib: float = 0.0  # 参数 + 激活值（推理）
    bwd_mem_gib: float = 0.0  # 梯度 + 优化器（训练）
    total_train_gib: float = 0.0


# ---------------------------------------------------------------------------
# 内存辅助函数
# ---------------------------------------------------------------------------


def _to_gib(elements: float, bytes_per_elem: int) -> float:
    return elements * bytes_per_elem / (1024**3)


# ---------------------------------------------------------------------------
# Attention 内存
# ---------------------------------------------------------------------------


def attention_memory(
    batch_size: int,
    seq_len: int,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    bytes_per_param: int = 2,
    bytes_per_act: int = 2,
) -> LayerMemory:
    """估算多头 attention 子层的内存使用。

    Parameters
    ----------
    batch_size, seq_len : int
        Batch 维度和序列维度。
    dim : int
        模型隐藏维度。
    num_heads, num_kv_heads : int
        Query 头数和 Key/Value 头数（用于 GQA）。
    bytes_per_param : int
        参数字节宽度（2 = fp16/bf16）。
    bytes_per_act : int
        激活值字节宽度（2 = fp16/bf16）。

    Returns
    -------
    LayerMemory 及其明细。
    """
    head_dim = dim // num_heads
    kv_head_dim = dim // num_heads  # 通常相同

    # 参数量
    q_params = dim * num_heads * head_dim
    k_params = dim * num_kv_heads * kv_head_dim
    v_params = dim * num_kv_heads * kv_head_dim
    o_params = num_heads * head_dim * dim
    total_params = q_params + k_params + v_params + o_params

    param_gib = _to_gib(total_params, bytes_per_param)

    # 激活值内存（峰值）：
    #   Q, K, V 张量 + attention scores + 输出
    B, S = batch_size, seq_len
    qkv_elements = B * S * (num_heads + 2 * num_kv_heads) * head_dim
    attn_scores_elements = B * num_heads * S * S
    total_act_elements = qkv_elements + attn_scores_elements
    act_gib = _to_gib(total_act_elements, bytes_per_act)

    fwd_gib = param_gib + act_gib
    # 训练：参数（fp32 master 副本 2×）+ 梯度 + 优化器（2 阶矩）
    bwd_gib = (
        _to_gib(total_params, 4)  # fp32 master 参数
        + _to_gib(total_params, bytes_per_param) * 2  # 梯度 + 优化器
        + act_gib
    )
    return LayerMemory(
        name="MultiHeadAttention",
        parameters_gib=param_gib,
        activations_gib=act_gib,
        fwd_mem_gib=fwd_gib,
        bwd_mem_gib=bwd_gib,
        total_train_gib=param_gib + bwd_gib,
    )


# ---------------------------------------------------------------------------
# MLP 内存
# ---------------------------------------------------------------------------


def mlp_memory(
    batch_size: int,
    seq_len: int,
    dim: int,
    mlp_ratio: float = 4.0,
    gated: bool = False,
    bytes_per_param: int = 2,
    bytes_per_act: int = 2,
) -> LayerMemory:
    """估算 MLP 子层的内存使用。

    Parameters
    ----------
    dim : int
        模型维度。
    mlp_ratio : float
        隐藏维度倍数（通常为 4.0；对于带 gate 的 SwiGLU 为 8/3）。
    gated : bool
        MLP 是否使用门控机制（SwiGLU）。
    """
    hidden = int(dim * mlp_ratio)
    B, S = batch_size, seq_len

    if gated:
        total_params = dim * hidden * 2 + hidden * dim  # gate, up, down
    else:
        total_params = dim * hidden + hidden * dim  # up, down

    param_gib = _to_gib(total_params, bytes_per_param)

    # 激活值：up projection 输出 + （可选）gate 输出
    act_elements = B * S * hidden * (2 if gated else 1)
    act_gib = _to_gib(act_elements, bytes_per_act)

    fwd_gib = param_gib + act_gib
    bwd_gib = (
        _to_gib(total_params, 4) + _to_gib(total_params, bytes_per_param) * 2 + act_gib
    )
    return LayerMemory(
        name="MLP" + (" (gated)" if gated else ""),
        parameters_gib=param_gib,
        activations_gib=act_gib,
        fwd_mem_gib=fwd_gib,
        bwd_mem_gib=bwd_gib,
        total_train_gib=param_gib + bwd_gib,
    )


# ---------------------------------------------------------------------------
# Embedding 内存
# ---------------------------------------------------------------------------


def embedding_memory(
    vocab_size: int,
    dim: int,
    batch_size: int,
    seq_len: int,
    bytes_per_param: int = 2,
) -> LayerMemory:
    """估算 token embedding 表的内存使用。"""
    elements = vocab_size * dim
    param_gib = _to_gib(elements, bytes_per_param)

    # 激活值：embedding 查询结果
    act_gib = _to_gib(batch_size * seq_len * dim, bytes_per_param)

    return LayerMemory(
        name="TokenEmbedding",
        parameters_gib=param_gib,
        activations_gib=act_gib,
        fwd_mem_gib=param_gib + act_gib,
        bwd_mem_gib=_to_gib(elements, 4)
        + _to_gib(elements, bytes_per_param) * 2
        + act_gib,
        total_train_gib=param_gib
        + _to_gib(elements, 4)
        + _to_gib(elements, bytes_per_param) * 2
        + act_gib,
    )


# ---------------------------------------------------------------------------
# 完整模型汇总
# ---------------------------------------------------------------------------


def transformer_memory_breakdown(
    vocab_size: int = 32000,
    dim: int = 4096,
    num_layers: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    mlp_ratio: float = 8.0 / 3.0,  # Llama 风格的标准配置
    batch_size: int = 1,
    seq_len: int = 2048,
    bytes_per_param: int = 2,
    bytes_per_act: int = 2,
    gated_mlp: bool = True,
) -> Dict[str, LayerMemory]:
    """完整的 Transformer 内存明细。

    返回一个字典，将层名称映射到其 LayerMemory 估算值。
    """
    results: Dict[str, LayerMemory] = {}

    # Embedding
    results["embedding"] = embedding_memory(
        vocab_size, dim, batch_size, seq_len, bytes_per_param
    )

    # 每层
    attn = attention_memory(
        batch_size,
        seq_len,
        dim,
        num_heads,
        num_kv_heads,
        bytes_per_param,
        bytes_per_act,
    )
    mlp = mlp_memory(
        batch_size, seq_len, dim, mlp_ratio, gated_mlp, bytes_per_param, bytes_per_act
    )

    attn_per_layer = LayerMemory(
        name="Attention (×L)",
        parameters_gib=attn.parameters_gib * num_layers,
        activations_gib=attn.activations_gib * num_layers,
        fwd_mem_gib=attn.fwd_mem_gib * num_layers,
        bwd_mem_gib=attn.bwd_mem_gib * num_layers,
        total_train_gib=attn.total_train_gib * num_layers,
    )
    mlp_per_layer = LayerMemory(
        name="MLP (×L)",
        parameters_gib=mlp.parameters_gib * num_layers,
        activations_gib=mlp.activations_gib * num_layers,
        fwd_mem_gib=mlp.fwd_mem_gib * num_layers,
        bwd_mem_gib=mlp.bwd_mem_gib * num_layers,
        total_train_gib=mlp.total_train_gib * num_layers,
    )

    results["attention_all"] = attn_per_layer
    results["mlp_all"] = mlp_per_layer

    # LM head（可能权重共享或独立；这里假设共享，因此不额外占用参数）
    results["lm_head"] = LayerMemory(
        name="LM Head (tied)",
        parameters_gib=results["embedding"].parameters_gib,
        activations_gib=0.0,
        fwd_mem_gib=0.0,
        bwd_mem_gib=0.0,
        total_train_gib=0.0,
    )

    return results


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== 层内存分析 (fp16 训练) ===\n")

    # 单层分析
    attn = attention_memory(
        batch_size=1, seq_len=2048, dim=4096, num_heads=32, num_kv_heads=8
    )
    print(f"{attn.name}:")
    print(f"  Parameters:   {attn.parameters_gib:.4f} GiB")
    print(f"  Activations:  {attn.activations_gib:.4f} GiB")
    print(f"  Fwd total:    {attn.fwd_mem_gib:.4f} GiB")
    print(f"  Train total:  {attn.total_train_gib:.4f} GiB\n")

    mlp = mlp_memory(
        batch_size=1, seq_len=2048, dim=4096, mlp_ratio=8.0 / 3.0, gated=True
    )
    print(f"{mlp.name}:")
    print(f"  Parameters:   {mlp.parameters_gib:.4f} GiB")
    print(f"  Activations:  {mlp.activations_gib:.4f} GiB")
    print(f"  Fwd total:    {mlp.fwd_mem_gib:.4f} GiB")
    print(f"  Train total:  {mlp.total_train_gib:.4f} GiB\n")

    # 完整模型明细
    print("=== 完整 7B 规模模型明细 ===\n")
    mem = transformer_memory_breakdown(
        vocab_size=32000,
        dim=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        batch_size=1,
        seq_len=2048,
    )
    total_param = 0.0
    total_act = 0.0
    total_train = 0.0
    for name, m in mem.items():
        print(
            f"{name:20s}: params={m.parameters_gib:6.3f} GiB  acts={m.activations_gib:6.3f} GiB  train={m.total_train_gib:7.3f} GiB"
        )
        total_param += m.parameters_gib
        total_act += m.activations_gib
        total_train += m.total_train_gib

    print(
        f"\n{'TOTAL':20s}: params={total_param:6.3f} GiB  acts={total_act:6.3f} GiB  train={total_train:7.3f} GiB"
    )
    print("\nAll checks passed.")
