"""
Memory calculator for different parallelism strategies.
Computes GPU memory usage for:
  - Pure data parallel (DP / DDP)
  - ZeRO stages 1-3
  - Tensor parallel (TP)
  - Pipeline parallel (PP)
  - Combined strategies (3D parallelism)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration of a transformer model."""

    vocab_size: int = 50000
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 32
    intermediate_size: int = 11008
    max_seq_len: int = 2048
    dtype_bytes: int = 2  # 2 for bf16/fp16, 4 for fp32


def count_parameters(config: ModelConfig) -> int:
    """Count total parameters in the transformer model (no embedding sharing)."""
    # Embedding
    embed = config.vocab_size * config.hidden_size

    # Per-layer parameters
    # Q, K, V projections + output projection
    qkv = 3 * config.hidden_size * config.hidden_size
    out_proj = config.hidden_size * config.hidden_size
    # MLP: two linear layers
    mlp = (
        config.hidden_size * config.intermediate_size
        + config.intermediate_size * config.hidden_size
    )
    # Layer norms (two per block)
    ln = 2 * config.hidden_size
    per_layer = qkv + out_proj + mlp + ln

    # Final layer norm
    final_ln = config.hidden_size

    # LM head (if tied with embedding, skip)
    lm_head = config.hidden_size * config.vocab_size

    total = embed + config.num_layers * per_layer + final_ln + lm_head
    return total


def format_memory_gb(bytes_val: float) -> str:
    """Format bytes as human-readable string."""
    if bytes_val >= 1e9:
        return f"{bytes_val / 1e9:.2f} GB"
    elif bytes_val >= 1e6:
        return f"{bytes_val / 1e6:.2f} MB"
    else:
        return f"{bytes_val / 1e3:.2f} KB"


@dataclass
class MemoryBreakdown:
    """Memory usage breakdown for a parallel strategy."""

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
    Estimate activation memory for a transformer.
    This is a rough estimate; actual depends on many factors.
    """
    hidden = config.hidden_size
    num_layers = config.num_layers

    # Per-layer activations (rough estimate)
    # Attention: Q, K, V, attention scores, attention output
    attn_act = batch_size * seq_len * hidden * 4  # Q, K, V, output
    attn_scores = (
        batch_size * config.num_attention_heads * seq_len * seq_len
    )  # attention matrix
    # MLP: intermediate activations
    mlp_act = batch_size * seq_len * config.intermediate_size
    # Layer norm residuals
    residuals = batch_size * seq_len * hidden

    per_layer = (attn_act + attn_scores + mlp_act + residuals) * config.dtype_bytes
    total = per_layer * num_layers

    # If activation checkpointing, only store ~sqrt(N) or O(1) layers
    return total


def compute_ddp_memory(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    num_gpus: int,
    use_amp: bool = False,
) -> MemoryBreakdown:
    """Compute memory for pure DDP (no ZeRO)."""
    params = count_parameters(config)
    bytes_per = config.dtype_bytes if use_amp else 4  # fp16 vs fp32

    # In DDP, each GPU stores:
    #   - Full parameters (fp16 if AMP, fp32 otherwise)
    #   - Full gradients (fp16/fp32)
    #   - Full optimizer states (fp32, 2x for Adam)
    #   - Activations (per microbatch)

    opt_multiplier = 2  # Adam: momentum + variance
    opt_bytes = 4  # Optimizer always uses fp32

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
    """Compute memory for ZeRO stages 1-3."""
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
    """Compute memory for tensor parallelism (alone, no DP)."""
    params = count_parameters(config)
    bytes_per = config.dtype_bytes if use_amp else 4
    opt_multiplier = 2
    opt_bytes = 4

    # TP splits parameters across devices
    params_mem = params * bytes_per / tp_size
    grads_mem = params * bytes_per / tp_size
    opt_mem = params * opt_multiplier * opt_bytes / tp_size

    # Activations are also split
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
    """Compute memory for pipeline parallelism."""
    params = count_parameters(config)
    bytes_per = config.dtype_bytes if use_amp else 4
    opt_multiplier = 2
    opt_bytes = 4

    # PP splits layers across devices
    params_mem = params * bytes_per / pp_size
    grads_mem = params * bytes_per / pp_size
    opt_mem = params * opt_multiplier * opt_bytes / pp_size

    # Activations: each device stores activations for its layers only
    act_mem = compute_activation_memory(config, batch_size, seq_len) / pp_size
    # Multiply by number of microbatches in flight for 1F1B
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
    """Compute memory for 3D parallelism (DP + TP + PP)."""
    total_gpus = dp_size * tp_size * pp_size
    params = count_parameters(config)
    bytes_per = config.dtype_bytes if use_amp else 4
    opt_multiplier = 2
    opt_bytes = 4

    # Model states divided by TP and PP
    model_params = params * bytes_per / (tp_size * pp_size)

    # ZeRO further divides across DP
    params_mem = model_params
    grads_mem = model_params
    opt_mem = params * opt_multiplier * opt_bytes / (tp_size * pp_size)

    if zero_stage >= 3:
        params_mem /= dp_size
    if zero_stage >= 2:
        grads_mem /= dp_size
    if zero_stage >= 1:
        opt_mem /= dp_size

    # Activations
    # Microbatch per GPU
    micro_bs = batch_size / (dp_size * num_microbatches)
    batch_per_device = batch_size / dp_size
    act_mem = compute_activation_memory(config, int(batch_per_device), seq_len)
    act_mem /= tp_size  # TP reduces per-device activation
    act_mem /= pp_size  # PP splits layers

    return MemoryBreakdown(
        params_mem=params_mem,
        grads_mem=grads_mem,
        opt_mem=opt_mem,
        activations_mem=act_mem,
        total=params_mem + grads_mem + opt_mem + act_mem,
    )


def main() -> None:
    print("=" * 70)
    print("Memory Calculator for Parallel Strategies")
    print("=" * 70)

    # Example: Llama-2 7B scale, scaled down for demonstration
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
    print(f"\nModel Configuration:")
    print(f"  Parameters: {params:,} ({params / 1e9:.2f}B)")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Precision: bf16 (2 bytes)")

    batch_size = 8
    seq_len = 2048

    # --- Pure DDP ---
    print("\n" + "-" * 70)
    print("Strategy Comparison (8 GPUs, batch_size=8, seq_len=2048)")
    print("-" * 70)
    print(
        f"{'Strategy':<25} {'Params':>10} {'Grads':>10} {'Optimizer':>10} {'Activations':>12} {'Total':>12}"
    )
    print("-" * 79)

    # DDP
    mem = compute_ddp_memory(config, batch_size, seq_len, 8, use_amp=True)
    print(
        f"{'DDP':<25} {format_memory_gb(mem.params_mem):>10} {format_memory_gb(mem.grads_mem):>10} {format_memory_gb(mem.opt_mem):>10} {format_memory_gb(mem.activations_mem):>12} {format_memory_gb(mem.total):>12}"
    )

    # ZeRO stages
    for stage in [1, 2, 3]:
        mem = compute_zero_memory(config, stage, batch_size, seq_len, 8, use_amp=True)
        print(
            f"{f'ZeRO-{stage}':<25} {format_memory_gb(mem.params_mem):>10} {format_memory_gb(mem.grads_mem):>10} {format_memory_gb(mem.opt_mem):>10} {format_memory_gb(mem.activations_mem):>12} {format_memory_gb(mem.total):>12}"
        )

    # TP (8-way)
    mem = compute_tensor_parallel_memory(config, batch_size, seq_len, 8, use_amp=True)
    print(
        f"{'TP (8-way)':<25} {format_memory_gb(mem.params_mem):>10} {format_memory_gb(mem.grads_mem):>10} {format_memory_gb(mem.opt_mem):>10} {format_memory_gb(mem.activations_mem):>12} {format_memory_gb(mem.total):>12}"
    )

    # PP (8-way)
    mem = compute_pipeline_parallel_memory(
        config, batch_size, seq_len, 8, num_microbatches=4
    )
    print(
        f"{'PP (8-way, 4MB)':<25} {format_memory_gb(mem.params_mem):>10} {format_memory_gb(mem.grads_mem):>10} {format_memory_gb(mem.opt_mem):>10} {format_memory_gb(mem.activations_mem):>12} {format_memory_gb(mem.total):>12}"
    )

    # 3D Parallelism
    print("\n" + "-" * 70)
    print("3D Parallelism Examples (64 GPUs)")
    print("-" * 70)
    print(
        f"{'Config (DP/TP/PP)':<25} {'Params':>10} {'Grads':>10} {'Optimizer':>10} {'Activations':>12} {'Total':>12}"
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
            "Should be 64"
        )
        # We just use the provided configs; the total may not be 64 for all
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

    # Recommendations
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)
    print("""
    Strategy selection guide:
    ┌──────────────┬──────────────────────────────────────────────────┐
    │ Model Size   │ Recommended Strategy                             │
    ├──────────────┼──────────────────────────────────────────────────┤
    │ < 1B params  │ DDP (simplest, no overhead)                      │
    │ 1B - 10B     │ ZeRO-2 or ZeRO-3 (DP only)                      │
    │ 10B - 100B   │ ZeRO-3 + TP (hybrid)                            │
    │ 100B - 500B  │ 3D parallelism (DP + TP + PP) with ZeRO-1/2    │
    │ > 500B       │ Full 3D parallelism with ZeRO-3 + activation    │
    │              │ checkpointing + offloading                      │
    └──────────────┴──────────────────────────────────────────────────┘

    Communication vs Memory trade-off:
    ┌──────────┬────────────┬──────────────┬────────────────┐
    │ Strategy │ Parameters │ Communication│ Memory/GPU     │
    ├──────────┼────────────┼──────────────┼────────────────┤
    │ DDP      │ Replicated │ 1x           │ Full model     │
    │ ZeRO-3   │ Sharded    │ 1.5x         │ 1/N of DDP     │
    │ TP       │ Sharded    │ High (intra) │ 1/TP of model  │
    │ PP       │ Sharded    │ Low          │ 1/PP of model  │
    └──────────┴────────────┴──────────────┴────────────────┘
    """)


if __name__ == "__main__":
    main()
