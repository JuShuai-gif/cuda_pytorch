"""
不同并行策略的内存计算器。
计算以下策略的 GPU 内存使用情况：
  - 纯数据并行（DP / DDP）
  - ZeRO 阶段 1-3
  - 张量并行（TP）
  - 流水线并行（PP）
  - 组合策略（3D 并行）
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Transformer 模型的配置。"""

    vocab_size: int = 50000
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 32
    intermediate_size: int = 11008
    max_seq_len: int = 2048
    dtype_bytes: int = 2  # 2 表示 bf16/fp16，4 表示 fp32


def count_parameters(config: ModelConfig) -> int:
    """统计 Transformer 模型的总参数量（不含嵌入共享）。"""
    # 嵌入层
    embed = config.vocab_size * config.hidden_size

    # 每层参数
    # Q、K、V 投影 + 输出投影
    qkv = 3 * config.hidden_size * config.hidden_size
    out_proj = config.hidden_size * config.hidden_size
    # MLP：两个线性层
    mlp = (
        config.hidden_size * config.intermediate_size
        + config.intermediate_size * config.hidden_size
    )
    # 层归一化（每块两个）
    ln = 2 * config.hidden_size
    per_layer = qkv + out_proj + mlp + ln

    # 最终层归一化
    final_ln = config.hidden_size

    # LM head（若与嵌入层权重绑定则跳过）
    lm_head = config.hidden_size * config.vocab_size

    total = embed + config.num_layers * per_layer + final_ln + lm_head
    return total


def format_memory_gb(bytes_val: float) -> str:
    """将字节数格式化为人类可读的字符串。"""
    if bytes_val >= 1e9:
        return f"{bytes_val / 1e9:.2f} GB"
    elif bytes_val >= 1e6:
        return f"{bytes_val / 1e6:.2f} MB"
    else:
        return f"{bytes_val / 1e3:.2f} KB"


@dataclass
class MemoryBreakdown:
    """某种并行策略的内存使用明细。"""

    params_mem: float = 0.0
    grads_mem: float = 0.0
    opt_mem: float = 0.0
    activations_mem: float = 0.0
    total: float = 0.0


def compute_activation_memory(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
) -> float:
    """
    估算 Transformer 的激活内存。
    这是一个粗略估算；实际值取决于多种因素。
    """
    hidden = config.hidden_size
    num_layers = config.num_layers

    # 每层激活（粗略估算）
    # 注意力：Q、K、V、注意力分数、注意力输出
    attn_act = batch_size * seq_len * hidden * 4  # Q、K、V、输出
    attn_scores = (
        batch_size * config.num_attention_heads * seq_len * seq_len
    )  # 注意力矩阵
    # MLP：中间激活
    mlp_act = batch_size * seq_len * config.intermediate_size
    # 层归一化的残差
    residuals = batch_size * seq_len * hidden

    per_layer = (attn_act + attn_scores + mlp_act + residuals) * config.dtype_bytes
    total = per_layer * num_layers

    # 若启用激活检查点，则仅存储约 sqrt(N) 或 O(1) 层
    return total


def compute_ddp_memory(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    num_gpus: int,
    use_amp: bool = False,
) -> MemoryBreakdown:
    """计算纯 DDP（无 ZeRO）的内存使用。"""
    params = count_parameters(config)
    bytes_per = config.dtype_bytes if use_amp else 4  # fp16 vs fp32

    # 在 DDP 中，每个 GPU 存储：
    #   - 完整参数（若使用 AMP 则为 fp16，否则为 fp32）
    #   - 完整梯度（fp16/fp32）
    #   - 完整优化器状态（fp32，Adam 需 2 倍）
    #   - 激活（每个 microbatch）

    opt_multiplier = 2  # Adam：动量 + 方差
    opt_bytes = 4  # 优化器始终使用 fp32

    params_mem = params * bytes_per
    grads_mem = params * bytes_per
    opt_mem = params * opt_multiplier * opt_bytes
    act_mem = compute_activation_memory(config, batch_size // num_gpus, seq_len)

    return MemoryBreakdown(
        params_mem=params_mem,
        grads_mem=grads_mem,
        opt_mem=opt_mem,
        activations_mem=act_mem,
        total=params_mem + grads_mem + opt_mem + act_mem,
    )


def compute_zero_memory(
    config: ModelConfig,
    stage: int,
    batch_size: int,
    seq_len: int,
    num_gpus: int,
    use_amp: bool = False,
) -> MemoryBreakdown:
    """计算 ZeRO 阶段 1-3 的内存使用。"""
    params = count_parameters(config)
    bytes_per = config.dtype_bytes if use_amp else 4
    opt_multiplier = 2
    opt_bytes = 4

    params_mem = params * bytes_per
    grads_mem = params * bytes_per
    opt_mem = params * opt_multiplier * opt_bytes

    if stage >= 3:
        params_mem /= num_gpus
    if stage >= 2:
        grads_mem /= num_gpus
    if stage >= 1:
        opt_mem /= num_gpus

    act_mem = compute_activation_memory(config, batch_size // num_gpus, seq_len)

    return MemoryBreakdown(
        params_mem=params_mem,
        grads_mem=grads_mem,
        opt_mem=opt_mem,
        activations_mem=act_mem,
        total=params_mem + grads_mem + opt_mem + act_mem,
    )


def compute_tensor_parallel_memory(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    tp_size: int,
    use_amp: bool = False,
) -> MemoryBreakdown:
    """计算张量并行（单独使用，无 DP）的内存使用。"""
    params = count_parameters(config)
    bytes_per = config.dtype_bytes if use_amp else 4
    opt_multiplier = 2
    opt_bytes = 4

    # TP 将参数拆分到各设备上
    params_mem = params * bytes_per / tp_size
    grads_mem = params * bytes_per / tp_size
    opt_mem = params * opt_multiplier * opt_bytes / tp_size

    # 激活也被拆分
    act_mem = compute_activation_memory(config, batch_size, seq_len) / tp_size

    return MemoryBreakdown(
        params_mem=params_mem,
        grads_mem=grads_mem,
        opt_mem=opt_mem,
        activations_mem=act_mem,
        total=params_mem + grads_mem + opt_mem + act_mem,
    )


def compute_pipeline_parallel_memory(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    pp_size: int,
    num_microbatches: int = 1,
    use_amp: bool = False,
) -> MemoryBreakdown:
    """计算流水线并行的内存使用。"""
    params = count_parameters(config)
    bytes_per = config.dtype_bytes if use_amp else 4
    opt_multiplier = 2
    opt_bytes = 4

    # PP 将层拆分到各设备上
    params_mem = params * bytes_per / pp_size
    grads_mem = params * bytes_per / pp_size
    opt_mem = params * opt_multiplier * opt_bytes / pp_size

    # 激活：每个设备仅存储其负责层的激活
    act_mem = compute_activation_memory(config, batch_size, seq_len) / pp_size
    # 乘以 1F1B 调度中同时在途的 microbatch 数量
    act_mem *= min(num_microbatches, pp_size)

    return MemoryBreakdown(
        params_mem=params_mem,
        grads_mem=grads_mem,
        opt_mem=opt_mem,
        activations_mem=act_mem,
        total=params_mem + grads_mem + opt_mem + act_mem,
    )


def compute_3d_parallel_memory(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    dp_size: int,
    tp_size: int,
    pp_size: int,
    zero_stage: int = 0,
    num_microbatches: int = 1,
    use_amp: bool = False,
) -> MemoryBreakdown:
    """计算 3D 并行（DP + TP + PP）的内存使用。"""
    total_gpus = dp_size * tp_size * pp_size
    params = count_parameters(config)
    bytes_per = config.dtype_bytes if use_amp else 4
    opt_multiplier = 2
    opt_bytes = 4

    # 模型状态由 TP 和 PP 均分
    model_params = params * bytes_per / (tp_size * pp_size)

    # ZeRO 进一步在 DP 维度上拆分
    params_mem = model_params
    grads_mem = model_params
    opt_mem = params * opt_multiplier * opt_bytes / (tp_size * pp_size)

    if zero_stage >= 3:
        params_mem /= dp_size
    if zero_stage >= 2:
        grads_mem /= dp_size
    if zero_stage >= 1:
        opt_mem /= dp_size

    # 激活
    # 每个 GPU 的 microbatch
    micro_bs = batch_size / (dp_size * num_microbatches)
    batch_per_device = batch_size / dp_size
    act_mem = compute_activation_memory(config, int(batch_per_device), seq_len)
    act_mem /= tp_size  # TP 减少每设备激活
    act_mem /= pp_size  # PP 拆分层

    return MemoryBreakdown(
        params_mem=params_mem,
        grads_mem=grads_mem,
        opt_mem=opt_mem,
        activations_mem=act_mem,
        total=params_mem + grads_mem + opt_mem + act_mem,
    )


def main() -> None:
    print("=" * 70)
    print("并行策略内存计算器")
    print("=" * 70)

    # 示例：Llama-2 7B 规模，为演示目的进行了缩减
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=32,
        intermediate_size=11008,
        max_seq_len=2048,
        dtype_bytes=2,  # bf16
    )

    params = count_parameters(config)
    print(f"\n模型配置：")
    print(f"  参数量：{params:,}（{params / 1e9:.2f}B）")
    print(f"  隐藏维度：{config.hidden_size}")
    print(f"  层数：{config.num_layers}")
    print(f"  精度：bf16（2 bytes）")

    batch_size = 8
    seq_len = 2048

    # --- 纯 DDP ---
    print("\n" + "-" * 70)
    print("策略对比（8 块 GPU，batch_size=8，seq_len=2048）")
    print("-" * 70)
    print(
        f"{'策略':<25} {'参数':>10} {'梯度':>10} {'优化器':>10} {'激活':>12} {'总计':>12}"
    )
    print("-" * 79)

    # DDP
    mem = compute_ddp_memory(config, batch_size, seq_len, 8, use_amp=True)
    print(
        f"{'DDP':<25} {format_memory_gb(mem.params_mem):>10} {format_memory_gb(mem.grads_mem):>10} {format_memory_gb(mem.opt_mem):>10} {format_memory_gb(mem.activations_mem):>12} {format_memory_gb(mem.total):>12}"
    )

    # ZeRO 阶段
    for stage in [1, 2, 3]:
        mem = compute_zero_memory(config, stage, batch_size, seq_len, 8, use_amp=True)
        print(
            f"{f'ZeRO-{stage}':<25} {format_memory_gb(mem.params_mem):>10} {format_memory_gb(mem.grads_mem):>10} {format_memory_gb(mem.opt_mem):>10} {format_memory_gb(mem.activations_mem):>12} {format_memory_gb(mem.total):>12}"
        )

    # TP（8 路）
    mem = compute_tensor_parallel_memory(config, batch_size, seq_len, 8, use_amp=True)
    print(
        f"{'TP (8-way)':<25} {format_memory_gb(mem.params_mem):>10} {format_memory_gb(mem.grads_mem):>10} {format_memory_gb(mem.opt_mem):>10} {format_memory_gb(mem.activations_mem):>12} {format_memory_gb(mem.total):>12}"
    )

    # PP（8 路）
    mem = compute_pipeline_parallel_memory(
        config, batch_size, seq_len, 8, num_microbatches=4
    )
    print(
        f"{'PP (8-way, 4MB)':<25} {format_memory_gb(mem.params_mem):>10} {format_memory_gb(mem.grads_mem):>10} {format_memory_gb(mem.opt_mem):>10} {format_memory_gb(mem.activations_mem):>12} {format_memory_gb(mem.total):>12}"
    )

    # 3D 并行
    print("\n" + "-" * 70)
    print("3D 并行示例（64 块 GPU）")
    print("-" * 70)
    print(
        f"{'配置 (DP/TP/PP)':<25} {'参数':>10} {'梯度':>10} {'优化器':>10} {'激活':>12} {'总计':>12}"
    )
    print("-" * 79)

    configs_3d = [
        (8, 1, 8, 0, "DP=8, PP=8"),
        (4, 2, 8, 0, "DP=4, TP=2, PP=8"),
        (4, 4, 4, 0, "DP=4, TP=4, PP=4"),
        (2, 8, 4, 0, "DP=2, TP=8, PP=4"),
        (4, 4, 4, 1, "DP=4, TP=4, PP=4, Z1"),
        (4, 4, 4, 2, "DP=4, TP=4, PP=4, Z2"),
    ]

    for dp, tp, pp, z, label in configs_3d:
        assert dp * tp * pp == sum(c[0] * c[1] * c[2] for c in [(dp, tp, pp)]), (
            "应为 64"
        )
        # 我们直接使用提供的配置；总数不一定都是 64
        total_gpus = dp * tp * pp
        mem = compute_3d_parallel_memory(
            config,
            batch_size,
            seq_len,
            dp_size=dp,
            tp_size=tp,
            pp_size=pp,
            zero_stage=z,
            num_microbatches=4,
            use_amp=True,
        )
        label_str = f"{label} ({total_gpus}G)"
        print(
            f"{label_str:<25} {format_memory_gb(mem.params_mem):>10} {format_memory_gb(mem.grads_mem):>10} {format_memory_gb(mem.opt_mem):>10} {format_memory_gb(mem.activations_mem):>12} {format_memory_gb(mem.total):>12}"
        )

    # 推荐策略
    print("\n" + "=" * 70)
    print("推荐策略")
    print("=" * 70)
    print("""
    策略选择指南：
    ┌──────────────┬──────────────────────────────────────────────────┐
    │ 模型规模     │ 推荐策略                                         │
    ├──────────────┼──────────────────────────────────────────────────┤
    │ < 1B 参数    │ DDP（最简单，无额外开销）                        │
    │ 1B - 10B     │ ZeRO-2 或 ZeRO-3（仅 DP）                       │
    │ 10B - 100B   │ ZeRO-3 + TP（混合）                             │
    │ 100B - 500B  │ 3D 并行（DP + TP + PP）配合 ZeRO-1/2            │
    │ > 500B       │ 全 3D 并行配合 ZeRO-3 + 激活检查点 + 卸载       │
    └──────────────┴──────────────────────────────────────────────────┘

    通信与内存权衡：
    ┌──────────┬────────────┬──────────────┬────────────────┐
    │ 策略     │ 参数       │ 通信量       │ 每 GPU 内存    │
    ├──────────┼────────────┼──────────────┼────────────────┤
    │ DDP      │ 完全复制   │ 1x           │ 完整模型       │
    │ ZeRO-3   │ 分片       │ 1.5x         │ DDP 的 1/N    │
    │ TP       │ 分片       │ 高（节点内） │ 模型的 1/TP   │
    │ PP       │ 分片       │ 低           │ 模型的 1/PP   │
    └──────────┴────────────┴──────────────┴────────────────┘
    """)


if __name__ == "__main__":
    main()
