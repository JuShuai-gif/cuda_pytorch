"""
Lab 02 解答: Resource Accounting & GPU

FLOPs、内存与 roofline 分析的完整实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_dim: int
    ffn_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    dtype_bytes: int = 2


@dataclass
class TrainingConfig:
    batch_size: int
    seq_len: int
    gradient_accumulation_steps: int = 1


# ══════════════════════════════════════════════════════════════════════
# 任务 1: FLOPs 计算
# ══════════════════════════════════════════════════════════════════════


def compute_transformer_flops(
    cfg: ModelConfig,
    tcfg: TrainingConfig,
    include_backward: bool = True,
) -> Dict[str, float]:
    d = cfg.hidden_dim
    d_ff = cfg.ffn_dim
    L = cfg.num_layers
    h = cfg.num_heads
    s = tcfg.seq_len
    B = tcfg.batch_size
    V = cfg.vocab_size

    # 每层每 token 的 FLOPs
    # ------------------------------------------------
    # QKV 投影:   (d, 3d)  matmul  -> 2 * d * 3d = 6d^2
    # Attention scores: (s, d/h) x (d/h, s) -> 2 * s * (d/h) * h * s = 2 s^2 d
    #   注意 — attention 的每 token FLOPs:
    #     Q @ K^T: s * d_h = s * (d/h) 每头, h 个头 -> s * d
    #     但 Q 是 (s, d), K 是 (s, d)。Q @ K^T 的输出是 (s, s)。
    #     FLOPs: 2 * s * d * s 每头? 不是:
    #     每头 (s, d_h) @ (d_h, s) = 2 * s * d_h * s = 2 s^2 d_h
    #     h 个头: 2 s^2 d_h * h = 2 s^2 d
    #
    #     Attention @ V: 每头 (s, s) @ (s, d_h) = 2 s^2 d_h
    #     h 个头: 2 s^2 d
    #
    #     所以 attention 总计 ≈ 4 s^2 d ... 但这是每句子的，不是每 token!
    #     不对 — 我们应该给出每 token 的 FLOPs。
    #
    # 更好的方法: 计算 batch 的总 FLOPs，然后除以 token 数
    # --------------------------------------------------------
    total_tokens = B * s

    # --- Embedding ---
    embed_flops = 0.0  # 可忽略，输入不是 matmul
    # lm_head: (B, s, d) @ (d, V) = 2 * B * s * d * V
    lm_head_flops = 2.0 * total_tokens * d * V

    # --- 每层 FLOPs ---
    # QKV: (B, s, d) @ (d, 3d) = 2 * B * s * d * 3d = 6 * B * s * d^2
    qkv_flops = 6.0 * total_tokens * d * d

    # Attention scores: Q @ K^T
    #   每头: (B, h, s, d_h) @ (B, h, d_h, s) -> (B, h, s, s)
    #   FLOPs: 2 * B * h * s * d_h * s = 2 * B * h * s^2 * d_h
    #   因为 h * d_h = d，所以 2 * B * s^2 * d
    attn_scores_flops = 2.0 * B * s * s * d

    # Attention @ V
    #   每头: (B, h, s, s) @ (B, h, s, d_h) -> (B, h, s, d_h)
    #   FLOPs: 2 * B * h * s * s * d_h = 2 * B * s^2 * d
    attn_value_flops = 2.0 * B * s * s * d

    # 输出投影: (B, s, d) @ (d, d) = 2 * B * s * d * d
    output_proj_flops = 2.0 * total_tokens * d * d

    # FFN gate: (B, s, d) @ (d, d_ff) = 2 * B * s * d * d_ff
    ffn_gate_flops = 2.0 * total_tokens * d * d_ff

    # FFN up: (B, s, d) @ (d, d_ff) = 2 * B * s * d * d_ff (for SwiGLU)
    ffn_up_flops = 2.0 * total_tokens * d * d_ff

    # FFN down: (B, s, d_ff) @ (d_ff, d) = 2 * B * s * d_ff * d
    ffn_down_flops = 2.0 * total_tokens * d_ff * d

    # --- 汇总 ---
    per_layer_flops = (
        qkv_flops
        + attn_scores_flops
        + attn_value_flops
        + output_proj_flops
        + ffn_gate_flops
        + ffn_up_flops
        + ffn_down_flops
    )
    total_forward = L * per_layer_flops + lm_head_flops

    # 反向约等于 2x 前向
    total_backward = 2 * total_forward if include_backward else 0.0

    return {
        "qkv_proj": L * qkv_flops,
        "attention": L * (attn_scores_flops + attn_value_flops),
        "output_proj": L * output_proj_flops,
        "ffn": L * (ffn_gate_flops + ffn_up_flops + ffn_down_flops),
        "embedding": lm_head_flops,
        "total_forward": total_forward,
        "total_backward": total_backward,
        "total": total_forward + total_backward,
    }


# ══════════════════════════════════════════════════════════════════════
# 任务 2: 内存核算
# ══════════════════════════════════════════════════════════════════════


def num_parameters(cfg: ModelConfig) -> int:
    """计算非 embedding 参数数量。"""
    d = cfg.hidden_dim
    d_ff = cfg.ffn_dim
    L = cfg.num_layers
    # 每层:
    #   QKV: d * 3d (无 bias)
    #   输出投影: d * d
    #   FFN gate: d * d_ff
    #   FFN up:   d * d_ff
    #   FFN down: d_ff * d
    #   LayerNorm x2: 2d * 2 (weight + bias)
    per_layer = (
        d * 3 * d  # QKV weight
        + d * d  # output proj weight
        + d * d_ff  # ffn gate
        + d * d_ff  # ffn up
        + d_ff * d  # ffn down
        + 2 * (2 * d)  # two LayerNorms
    )
    N = L * per_layer
    # 最终 LayerNorm
    N += 2 * d
    # lm_head (与 embedding 共享或独立)
    N += d * cfg.vocab_size
    return N


def compute_memory_breakdown(
    cfg: ModelConfig,
    tcfg: TrainingConfig,
    use_mixed_precision: bool = True,
    optimizer: str = "adam",
) -> Dict[str, float]:
    d = cfg.hidden_dim
    d_ff = cfg.ffn_dim
    L = cfg.num_layers
    s = tcfg.seq_len
    B = tcfg.batch_size
    V = cfg.vocab_size
    N = num_parameters(cfg)

    param_bytes = 2 if use_mixed_precision else 4
    grad_bytes = 2 if use_mixed_precision else 4
    opt_bytes = 4  # Adam 状态始终为 FP32

    memory_params = N * param_bytes
    memory_grads = N * grad_bytes

    if optimizer.lower() == "adam":
        # Adam 存储 m 和 v，均为 FP32
        memory_optimizer = N * opt_bytes * 2
    else:
        memory_optimizer = N * opt_bytes  # SGD momentum (FP32)

    # 激活内存估算（粗略）
    # 每个 Transformer block: QKV 输出 + attention 输出 + FFN 隐藏状态 + 残差
    # 每个激活: ~B * s * d * param_bytes
    activation_per_block = B * s * d * param_bytes * 6  # 粗略估算
    memory_activations = L * activation_per_block

    # KV Cache (在推理时使用，而非训练 — 但为完整性包含)
    # 每层: 2 (K+V) * s * d * param_bytes * B
    memory_kv_cache = L * 2 * s * d * param_bytes * B

    total = memory_params + memory_grads + memory_optimizer + memory_activations

    return {
        "parameters": memory_params,
        "gradients": memory_grads,
        "optimizer_states": memory_optimizer,
        "activations": memory_activations,
        "kv_cache": memory_kv_cache,
        "total": total,
        "total_gb": total / (1024**3),
    }


# ══════════════════════════════════════════════════════════════════════
# 任务 3: 算术强度与 Roofline 模型
# ══════════════════════════════════════════════════════════════════════


def arithmetic_intensity(flops: float, bytes_read: float, bytes_write: float) -> float:
    return flops / (bytes_read + bytes_write)


def roofline_classification(
    ai: float,
    peak_flops: float,
    peak_bandwidth: float,
) -> str:
    ridge_point = peak_flops / peak_bandwidth
    return "compute-bound" if ai >= ridge_point else "memory-bound"


def roofline_attainable_performance(
    ai: float,
    peak_flops: float,
    peak_bandwidth: float,
) -> float:
    return min(peak_flops, ai * peak_bandwidth)


# ══════════════════════════════════════════════════════════════════════
# 任务 4: PyTorch Profiler
# ══════════════════════════════════════════════════════════════════════


def profile_simple_transformer(
    cfg: ModelConfig,
    seq_len: int,
    batch_size: int,
    num_steps: int = 5,
) -> None:
    model = _build_mini_transformer(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    # 预热
    for _ in range(3):
        _ = model(x)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_steps):
            y = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace("/tmp/transformer_trace.json")
    print("Chrome trace 已保存至 /tmp/transformer_trace.json")


def _build_mini_transformer(cfg: ModelConfig) -> nn.Module:
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


# ══════════════════════════════════════════════════════════════════════
# 验证
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # LLaMA-7B 风格配置（为演示缩小规模）
    cfg = ModelConfig(
        vocab_size=32000,
        hidden_dim=4096,
        ffn_dim=11008,
        num_layers=32,
        num_heads=32,
        head_dim=128,
        max_seq_len=2048,
    )
    tcfg = TrainingConfig(batch_size=1, seq_len=2048)

    print("=== Lab 02 解答验证 ===\n")

    # --- FLOPs ---
    flops = compute_transformer_flops(cfg, tcfg)
    print("FLOPs 分解 (单次迭代):")
    for k, v in flops.items():
        if v > 1e12:
            print(f"  {k:20s}: {v / 1e12:.2f} TFLOPs")
        elif v > 1e9:
            print(f"  {k:20s}: {v / 1e9:.2f} GFLOPs")
        else:
            print(f"  {k:20s}: {v:.2f} FLOPs")

    # --- 参数数量 ---
    N = num_parameters(cfg)
    print(f"\n参数总量: {N / 1e6:.1f}M (非 embedding: {N / 1e9:.2f}B)")

    # --- 内存 ---
    mem = compute_memory_breakdown(cfg, tcfg)
    print(f"\n内存分解:")
    for k, v in mem.items():
        if k == "total_gb":
            print(f"  {k:20s}: {v:.2f} GB")
        elif v > 1e9:
            print(f"  {k:20s}: {v / 1e9:.2f} GB")
        else:
            print(f"  {k:20s}: {v / 1e6:.2f} MB")

    # --- Roofline ---
    ai_matmul = arithmetic_intensity(
        flops=2 * cfg.hidden_dim**2 * tcfg.seq_len,
        bytes_read=cfg.hidden_dim**2 * 2,  # FP16
        bytes_write=cfg.hidden_dim**2 * 2,
    )
    peak_a100 = 312e12  # A100 FP16 TFLOPS
    bw_a100 = 2.039e12  # A100 HBM 带宽 bytes/s
    classification = roofline_classification(ai_matmul, peak_a100, bw_a100)
    perf = roofline_attainable_performance(ai_matmul, peak_a100, bw_a100)
    print(f"\nMatmul 的 Roofline 分析 (AI={ai_matmul:.1f} FLOP/byte):")
    print(f"  分类: {classification}")
    print(f"  可达: {perf / 1e12:.1f} TFLOPS")

    # --- Profiler (仅当 CUDA 可用时) ---
    if torch.cuda.is_available():
        print("\n运行 profiler (5 步)...")
        profile_simple_transformer(cfg, seq_len=128, batch_size=2, num_steps=5)
    else:
        print("\n无 CUDA 设备 — 跳过 profiler 演示。")
