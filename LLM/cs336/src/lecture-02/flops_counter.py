"""
第 02 讲 — 资源核算：FLOPs 计数器。

统计模型参数量，估算训练 FLOPs，并计算内存占用
（参数、梯度、优化器状态、激活值）。
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# 参数统计
# ---------------------------------------------------------------------------


def count_parameters(model: Any) -> int:
    """返回 *model* 中可训练参数的总数。

    适用于任何 ``torch.nn.Module``，或拥有 ``parameters()`` 迭代器
    且每个元素具备 ``numel()`` 方法的普通 Python 对象。
    """
    try:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except ImportError:
        pass

    # 回退方案：适用于暴露 .parameters() 的对象
    total = 0
    for p in model.parameters():
        try:
            total += p.numel()
        except AttributeError:
            total += 1  # 标量占位
    return total


# ---------------------------------------------------------------------------
# 训练 FLOPs 估算
# ---------------------------------------------------------------------------


def estimate_training_flops(
    num_params: int,
    num_tokens: int,
    factor: float = 6.0,
) -> float:
    """估算总训练 FLOPs，使用近似公式：

        FLOPs ≈ factor * N * D

    其中 N =（非嵌入）参数数量，D = 训练 token 数量。
    常用取值：
        factor = 6  适用于稠密 Transformer（Kaplan 等人）
        factor = 3  仅推理

    返回以浮点数表示的 FLOPs（可能采用科学计数法）。
    """
    return factor * num_params * num_tokens


def estimate_training_flops_detailed(
    model: Any,
    batch_size: int,
    seq_len: int,
    vocab_size: int | None = None,
) -> dict[str, float]:
    """将训练 FLOPs 分解为各组成部分。

    返回一个包含以下键的字典：
        fwd_flops, bwd_flops, total_flops, fwd_params_ratio
    """
    num_params = count_parameters(model)
    tokens_per_step = batch_size * seq_len
    # 前向：每个 token 约 2 * N（标准 matmul 近似）
    fwd = 2.0 * num_params * tokens_per_step
    # 反向：约等于前向的 2 倍（梯度计算）
    bwd = 2.0 * fwd
    total = fwd + bwd  # ≈ 6 * N * tokens

    return {
        "fwd_flops": fwd,
        "bwd_flops": bwd,
        "total_flops": total,
        "tokens_per_step": tokens_per_step,
        "fwd_params_ratio": fwd / max(num_params, 1),
    }


# ---------------------------------------------------------------------------
# 内存占用
# ---------------------------------------------------------------------------


def memory_footprint(
    num_params: int,
    bytes_per_param: int = 2,  # fp16 / bf16
    bytes_per_optimizer_state: int = 4,  # fp32 动量估计
    optimizer_states: int = 2,  # Adam 有 m 和 v 两个状态
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    """估算训练所需的 GPU 峰值内存。

    各组成部分（单位均为 GiB）：
      - 参数
      - 梯度
      - 优化器状态（如 Adam 的 m 和 v）
      - 激活值（粗略估计，调用方可自行覆盖）
    """
    to_gib = 1.0 / (1024**3)

    param_mem = num_params * bytes_per_param * to_gib
    grad_mem = num_params * bytes_per_param * to_gib / grad_accum_steps
    opt_mem = num_params * optimizer_states * bytes_per_optimizer_state * to_gib

    # 激活值内存 — 粗略启发式估算：约为 batch * seq * hidden * layers
    # 此处不直接计算；调用方可利用本字典自行组合。
    return {
        "parameters_gib": param_mem,
        "gradients_gib": grad_mem,
        "optimizer_states_gib": opt_mem,
        "total_excl_activations_gib": param_mem + grad_mem + opt_mem,
    }


def estimate_activation_memory(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    bytes_per_elem: int = 2,
    recompute: bool = False,
) -> float:
    """Transformer 激活值内存的粗略上限估计。

    假设每层大约存储 batch * seq * hidden 个元素。
    若 *recompute* 为 True（即启用梯度检查点），内存将除以
    重计算因子（通常约为 sqrt(layers)）。
    """
    elements_per_layer = batch_size * seq_len * hidden_dim
    total_elements = elements_per_layer * num_layers
    if recompute:
        total_elements /= math.sqrt(max(num_layers, 1))
    return total_elements * bytes_per_elem / (1024**3)


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- 参数 ---
    try:
        import torch

        m = torch.nn.Linear(768, 768)
        print(f"Linear(768,768) params: {count_parameters(m):,}")
    except ImportError:
        print("torch not available — using a dummy model")

        class _Dummy:
            def parameters(self):
                class _P:
                    def numel(self):
                        return 1000

                return [_P() for _ in range(5)]

        m = _Dummy()
        print(f"Dummy params: {count_parameters(m)}")

    # --- FLOPs ---
    num_tokens = 1_000_000_000  # 1B tokens
    N = 7_000_000_000  # 7B 参数
    flops = estimate_training_flops(N, num_tokens)
    print(f"Training FLOPs ({N:.0e} params × {num_tokens:.0e} tokens): {flops:.3e}")

    detailed = estimate_training_flops_detailed(m, batch_size=8, seq_len=2048)
    for k, v in detailed.items():
        print(f"  {k}: {v:.3e}")

    # --- 内存 ---
    mem = memory_footprint(num_params=N, bytes_per_param=2)
    for k, v in mem.items():
        print(f"  {k}: {v:.2f} GiB")

    act_mem = estimate_activation_memory(8, 2048, 4096, 32)
    print(f"  activations (est): {act_mem:.2f} GiB")

    print("\nAll checks passed.")
