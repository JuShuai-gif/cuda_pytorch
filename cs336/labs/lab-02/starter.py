"""
Lab 02: Resource Accounting & GPU — 起始代码

完成以下占位实现:
  - compute_transformer_flops
  - compute_memory_breakdown
  - arithmetic_intensity
  - roofline_classification
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Transformer 模型配置。"""

    vocab_size: int  # V
    hidden_dim: int  # d
    ffn_dim: int  # d_ff
    num_layers: int  # L
    num_heads: int  # h
    head_dim: int  # d_h = d / h
    max_seq_len: int  # s_max
    dtype_bytes: int = 2  # FP16 = 2, FP32 = 4


@dataclass
class TrainingConfig:
    """训练超参数。"""

    batch_size: int
    seq_len: int
    gradient_accumulation_steps: int = 1


# ──────────────────────────────────────────────────────────────────────
# 任务 1: FLOPs 计算
# ──────────────────────────────────────────────────────────────────────


def compute_transformer_flops(
    cfg: ModelConfig,
    tcfg: TrainingConfig,
    include_backward: bool = True,
) -> Dict[str, float]:
    """计算 Transformer 前向 + 反向的 FLOPs 分解。

    Args:
        cfg: 模型配置。
        tcfg: 训练配置（用于按迭代次数缩放）。
        include_backward: 如果为 True，估算总量（前向 + 反向）。

    Returns:
        字典，包含键: 'qkv_proj', 'attention', 'output_proj',
                      'ffn', 'embedding', 'total_forward',
                      'total_backward', 'total'。

    所有值单位为 FLOPs（而非 MACs）。1 MAC = 2 FLOPs。
    """
    d = cfg.hidden_dim
    d_ff = cfg.ffn_dim
    L = cfg.num_layers
    s = tcfg.seq_len
    B = tcfg.batch_size
    V = cfg.vocab_size

    # TODO: 实现 FLOPs 计算
    raise NotImplementedError("compute_transformer_flops() not implemented")


# ──────────────────────────────────────────────────────────────────────
# 任务 2: 内存核算
# ──────────────────────────────────────────────────────────────────────


def compute_memory_breakdown(
    cfg: ModelConfig,
    tcfg: TrainingConfig,
    use_mixed_precision: bool = True,
    optimizer: str = "adam",
) -> Dict[str, float]:
    """估算训练时的 GPU 显存使用量。

    返回一个字典，以字节为单位包含:
      - 'parameters'
      - 'gradients'
      - 'optimizer_states'   (Adam 的 m+v)
      - 'activations'        (每层峰值 * num_layers，粗略估算)
      - 'kv_cache'           (如适用)
      - 'total'

    Args:
        cfg: 模型配置。
        tcfg: 训练配置。
        use_mixed_precision: 如果为 True，params/grads 为 FP16，optimizer 为 FP32。
        optimizer: 'adam' 或 'sgd'。
    """
    # TODO: 实现内存分解
    raise NotImplementedError("compute_memory_breakdown() not implemented")


def num_parameters(cfg: ModelConfig) -> int:
    """计算参数数量（不含 embedding）。"""
    # TODO: 计算参数数量
    raise NotImplementedError("num_parameters() not implemented")


# ──────────────────────────────────────────────────────────────────────
# 任务 3: 算术强度与 Roofline 模型
# ──────────────────────────────────────────────────────────────────────


def arithmetic_intensity(flops: float, bytes_read: float, bytes_write: float) -> float:
    """计算算术强度 = FLOPs / 总传输字节数。

    Args:
        flops: 浮点运算次数。
        bytes_read: 从内存读取的字节数。
        bytes_write: 写入内存的字节数。

    Returns:
        算术强度 (FLOPs/byte)。
    """
    # TODO: 实现算术强度计算
    raise NotImplementedError("arithmetic_intensity() not implemented")


def roofline_classification(
    ai: float,
    peak_flops: float,
    peak_bandwidth: float,
) -> str:
    """将操作分类为计算受限(compute-bound)或内存受限(memory-bound)。

    Args:
        ai: 算术强度。
        peak_flops: GPU 峰值 FLOPS/s（如 A100 FP16 为 312e12）。
        peak_bandwidth: GPU 峰值 HBM 带宽，单位 bytes/s（如 A100 为 2.039e12）。

    Returns:
        'compute-bound' 或 'memory-bound'。
    """
    # TODO: 实现分类
    raise NotImplementedError("roofline_classification() not implemented")


def roofline_attainable_performance(
    ai: float,
    peak_flops: float,
    peak_bandwidth: float,
) -> float:
    """在 roofline 模型下计算可达性能。

    attainable = min(peak_flops, ai * peak_bandwidth)

    Args:
        ai: 算术强度。
        peak_flops: 峰值计算能力 (FLOPS/s)。
        peak_bandwidth: 峰值带宽 (bytes/s)。

    Returns:
        可达 FLOPS/s。
    """
    # TODO: 实现 roofline 性能计算
    raise NotImplementedError("roofline_attainable_performance() not implemented")


# ──────────────────────────────────────────────────────────────────────
# 任务 4: PyTorch Profiler（集成桩代码）
# ──────────────────────────────────────────────────────────────────────


def profile_simple_transformer(
    cfg: ModelConfig,
    seq_len: int,
    batch_size: int,
    num_steps: int = 5,
) -> None:
    """在简单 Transformer 上前向传播时运行 PyTorch profiler。

    使用 torch.profiler 完成:
    1. 记录 CPU/GPU 活动
    2. 打印汇总表
    3. 导出 Chrome trace
    """
    # 构建一个最小的 Transformer
    model = _build_mini_transformer(cfg)

    # TODO: 实现 profiling
    raise NotImplementedError("profile_simple_transformer() not implemented")


def _build_mini_transformer(cfg: ModelConfig) -> nn.Module:
    """构建用于 profiling 的最小 Transformer 模型。"""

    class MiniBlock(nn.Module):
        def __init__(self, d: int, d_ff: int, h: int):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, h, batch_first=True)
            self.ln1 = nn.LayerNorm(d)
            self.ln2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
            x = x + a
            x = x + self.ffn(self.ln2(x))
            return x

    class MiniTransformer(nn.Module):
        def __init__(self, cfg: ModelConfig):
            super().__init__()
            self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
            self.layers = nn.ModuleList(
                [
                    MiniBlock(cfg.hidden_dim, cfg.ffn_dim, cfg.num_heads)
                    for _ in range(cfg.num_layers)
                ]
            )
            self.ln_f = nn.LayerNorm(cfg.hidden_dim)
            self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            x = self.ln_f(x)
            return self.lm_head(x)

    return MiniTransformer(cfg)


if __name__ == "__main__":
    print("Lab 02 starter — 运行 'python solution.py' 验证。")
