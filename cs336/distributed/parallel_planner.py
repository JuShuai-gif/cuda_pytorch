"""
Parallel Strategy Planner for distributed training.

Implements:
- Model specification and hardware configuration
- Memory estimation for all parallelism strategies (DP, ZeRO, TP, PP, DP+TP+PP)
- Communication overhead estimation
- MFU (Model FLOPs Utilization) computation
- PTD-P (Pipeline + Tensor + Data Parallelism) configuration optimization
- Automatic strategy recommendation based on model and hardware constraints

Key formulas:
- Total memory = params_mem + grads_mem + optimizer_mem + activation_mem
- MFU = actual_flops / peak_flops where peak_flops = FLOPS_per_GPU * num_GPUs
- Ring All-Reduce time = 2 * data_size * (P-1) / (P * bandwidth)
- Bubble ratio = (P-1) / M for GPipe
- TP communication = 4 * All-Reduce per transformer layer per step

Hardware reference numbers:
- H100 SXM: 989 TFLOPS (fp16), 80 GB HBM, 3350 GB/s HBM bandwidth, NVLink 900 GB/s
- A100 SXM: 312 TFLOPS (fp16), 80 GB HBM, 2039 GB/s HBM bandwidth, NVLink 600 GB/s
- InfiniBand NDR400: 400 GB/s per port
- PCIe 5.0: 64 GB/s
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Hardware configuration
# ---------------------------------------------------------------------------


@dataclass
class HardwareConfig:
    """
    Hardware configuration for distributed training.

    Default values represent an 8x H100 SXM node.

    Attributes:
        gpu_model: GPU model identifier.
        num_gpus: Total number of GPUs available.
        gpus_per_node: GPUs per node (controls NVLink vs IB split).
        gpu_memory_gb: GPU HBM capacity (e.g., 80 GB for H100).
        peak_flops_tflops: Peak compute in TFLOPS (fp16 tensor core).
        memory_bandwidth_gb_s: HBM memory bandwidth.
        intra_node_bandwidth_gb_s: NVLink/NVSwitch bandwidth.
        inter_node_bandwidth_gb_s: InfiniBand/RoCE bandwidth.
        pcie_bandwidth_gb_s: PCIe bandwidth for CPU-GPU transfers.
        supports_fp8: Whether GPU supports FP8 tensor cores.
        supports_nvlink: Whether GPUs are connected via NVLink.
    """

    gpu_model: str = "H100-SXM"
    num_gpus: int = 8
    gpus_per_node: int = 8
    gpu_memory_gb: float = 80.0
    peak_flops_tflops: float = 989.0  # fp16 tensor core
    memory_bandwidth_gb_s: float = 3350.0
    intra_node_bandwidth_gb_s: float = 900.0
    inter_node_bandwidth_gb_s: float = 400.0
    pcie_bandwidth_gb_s: float = 64.0
    supports_fp8: bool = True
    supports_nvlink: bool = True

    @property
    def num_nodes(self) -> int:
        return max(1, math.ceil(self.num_gpus / self.gpus_per_node))

    @staticmethod
    def h100_8gpu() -> HardwareConfig:
        """Standard 8x H100 SXM node."""
        return HardwareConfig(
            gpu_model="H100-SXM",
            num_gpus=8,
            gpus_per_node=8,
            gpu_memory_gb=80.0,
            peak_flops_tflops=989.0,
            memory_bandwidth_gb_s=3350.0,
            intra_node_bandwidth_gb_s=900.0,
            inter_node_bandwidth_gb_s=400.0,
        )

    @staticmethod
    def a100_8gpu() -> HardwareConfig:
        """Standard 8x A100 SXM node."""
        return HardwareConfig(
            gpu_model="A100-SXM",
            num_gpus=8,
            gpus_per_node=8,
            gpu_memory_gb=80.0,
            peak_flops_tflops=312.0,
            memory_bandwidth_gb_s=2039.0,
            intra_node_bandwidth_gb_s=600.0,
            inter_node_bandwidth_gb_s=200.0,
        )


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """
    Transformer model specification for memory and FLOPs estimation.

    Supports Gemma, Llama, and custom model architectures.

    Attributes:
        name: Model identifier.
        vocab_size: Vocabulary size.
        hidden_size: Hidden dimension (d_model).
        num_layers: Number of transformer layers.
        num_attention_heads: Number of query heads.
        num_kv_heads: Number of KV heads (for GQA; equals num_attention_heads for MHA).
        intermediate_size: Feed-forward intermediate size.
        max_seq_len: Maximum sequence length.
        dtype_bytes: Bytes per parameter (2 for bf16/fp16, 1 for fp8, 4 for fp32).
        opt_state_bytes: Bytes per optimizer state element (4 for fp32 Adam).
        opt_state_multiplier: Multiplier for optimizer states (2 for Adam m+v).
        activation_multiplier: Rough multiplier for activation memory estimation.
    """

    name: str = "custom"
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 32
    intermediate_size: int = 11008
    max_seq_len: int = 4096
    dtype_bytes: int = 2  # bf16
    opt_state_bytes: int = 4  # fp32
    opt_state_multiplier: int = 2  # Adam: m + v
    activation_multiplier: float = 1.0

    @staticmethod
    def llama_7b() -> ModelSpec:
        """Llama-2 7B model specification."""
        return ModelSpec(
            name="Llama-2-7B",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=32,
            intermediate_size=11008,
            max_seq_len=4096,
        )

    @staticmethod
    def llama_70b() -> ModelSpec:
        """Llama-2 70B model specification."""
        return ModelSpec(
            name="Llama-2-70B",
            vocab_size=32000,
            hidden_size=8192,
            num_layers=80,
            num_attention_heads=64,
            num_kv_heads=8,
            intermediate_size=28672,
            max_seq_len=4096,
        )

    @staticmethod
    def gemma_7b() -> ModelSpec:
        """Gemma-7B model specification."""
        return ModelSpec(
            name="Gemma-7B",
            vocab_size=256000,
            hidden_size=3072,
            num_layers=28,
            num_attention_heads=16,
            num_kv_heads=16,
            intermediate_size=24576,
            max_seq_len=8192,
        )


# ---------------------------------------------------------------------------
# Parallel configuration
# ---------------------------------------------------------------------------


@dataclass
class ParallelConfig:
    """
    PTD-P parallelism configuration.

    Data (D), Tensor (T), Pipeline (P) parallelism dimensions,
    plus ZeRO stage for data parallelism memory optimization.

    Constraints:
    - tp_size * pp_size * dp_size = total_gpus
    - tp_size <= gpus_per_node (NVLink domain, typically ≤ 8)
    - pp_size >= 1 (often = num_nodes for cross-node PP)
    - dp_size >= 1 (can be very large with ZeRO)
    """

    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    zero_stage: int = 0
    num_microbatches: int = 1
    sequence_parallel: bool = False
    expert_parallel_size: int = 1
    activation_checkpointing: bool = False

    @property
    def total_gpus(self) -> int:
        return self.tp_size * self.pp_size * self.dp_size


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


@dataclass
class MemoryEstimate:
    """Memory usage estimate for a single GPU (in GB)."""

    params_gb: float = 0.0
    grads_gb: float = 0.0
    optimizer_gb: float = 0.0
    activations_gb: float = 0.0
    total_gb: float = 0.0
    peak_gb: float = 0.0

    def __post_init__(self) -> None:
        self.total_gb = (
            self.params_gb + self.grads_gb + self.optimizer_gb + self.activations_gb
        )
        self.peak_gb = self.total_gb * 1.1  # ~10% overhead buffer


def count_parameters(model: ModelSpec) -> int:
    """Count total non-embedding and total parameters in a transformer.

    Returns total parameter count including embeddings and LM head.
    """
    # Embedding
    embed_params = model.vocab_size * model.hidden_size

    # Per-layer parameters
    # Attention: Q, K, V, O projections
    q_proj = model.hidden_size * model.hidden_size
    k_proj = model.hidden_size * (
        model.num_kv_heads * (model.hidden_size // model.num_attention_heads)
    )
    v_proj = k_proj  # Same shape as K
    o_proj = model.hidden_size * model.hidden_size
    attn_params = q_proj + k_proj + v_proj + o_proj

    # MLP: two linear layers (SwiGLU uses 3 for gate)
    # Standard: fc1 (hidden → intermediate) + fc2 (intermediate → hidden)
    mlp_params = model.hidden_size * model.intermediate_size * 2  # fc1 + fc2

    # Layer norm: 2 per layer (pre-attn, pre-mlp)
    ln_params = 2 * model.hidden_size * 2  # weight + bias

    # Per-layer total
    per_layer_params = attn_params + mlp_params + ln_params

    # Final layer norm + LM head
    final_ln_params = model.hidden_size * 2
    lm_head_params = model.hidden_size * model.vocab_size

    total = (
        embed_params
        + model.num_layers * per_layer_params
        + final_ln_params
        + lm_head_params
    )
    return int(total)


def estimate_memory(
    model: ModelSpec,
    parallel: ParallelConfig,
    batch_size: int,
    seq_len: int,
    hardware: HardwareConfig,
) -> MemoryEstimate:
    """
    Estimate per-GPU memory usage for a given parallel configuration.

    Computation follows the 16Ψ formula:
    - Total memory = params + grads + optimizer_states + activations
    - Each component is divided by relevant parallelism dimensions

    For mixed precision (bf16 params, fp32 optimizer):
    - params_mem = total_params * 2 / (tp * pp * (dp if zero>=3))
    - grads_mem = total_params * 2 / (tp * pp * (dp if zero>=2))
    - opt_mem = total_params * 4 * multiplier / (tp * pp * (dp if zero>=1))
    - act_mem = roughly based on batch, seq, hidden, layers

    Args:
        model: Model specification.
        parallel: Parallel configuration.
        batch_size: Global batch size.
        seq_len: Sequence length.
        hardware: Hardware configuration.

    Returns:
        MemoryEstimate with per-GPU memory breakdown.
    """
    total_params = count_parameters(model)

    tp = parallel.tp_size
    pp = parallel.pp_size
    dp = parallel.dp_size
    zs = parallel.zero_stage

    # Effective DP size after excluding TP and PP
    effective_dp = dp

    # Compute per-GPU parameter-related memory
    # Model parallelism (TP + PP) divides the model
    model_parallel_factor = tp * pp

    params_bytes = total_params * model.dtype_bytes
    grads_bytes = total_params * model.dtype_bytes
    opt_bytes = total_params * model.opt_state_bytes * model.opt_state_multiplier

    # Apply sharding
    params_gb = params_bytes / model_parallel_factor
    grads_gb = grads_bytes / model_parallel_factor
    opt_gb = opt_bytes / model_parallel_factor

    if zs >= 3:
        params_gb /= effective_dp
    if zs >= 2:
        grads_gb /= effective_dp
    if zs >= 1:
        opt_gb /= effective_dp

    # Convert to GB
    params_gb /= 1e9
    grads_gb /= 1e9
    opt_gb /= 1e9

    # Activation memory estimation
    # Rough formula: activation ~ batch_per_gpu * seq * hidden * num_layers * dtype * 10-30x factor
    micro_batch = (
        batch_size / (dp * parallel.num_microbatches) if dp > 0 else batch_size
    )
    layers_per_gpu = model.num_layers / pp if pp > 0 else model.num_layers

    # Activation memory per microbatch per layer (very rough estimate)
    # A single transformer layer's activations ~ O(batch * seq * hidden) * factor
    tokens_per_microbatch = micro_batch * seq_len
    act_per_layer = (
        tokens_per_microbatch
        * model.hidden_size
        * model.dtype_bytes
        * 10  # Empirical factor for QKV, scores, MLP intermediate, etc.
    )

    # With 1F1B or GPipe, concurrent in-flight microbatches varies
    if parallel.num_microbatches > 1:
        # 1F1B: peak ≈ min(pp, num_microbatches) worth of activations
        active_microbatches = min(pp, parallel.num_microbatches)
        act_mem = act_per_layer * layers_per_gpu * active_microbatches
    else:
        act_mem = act_per_layer * layers_per_gpu

    act_gb = act_mem / 1e9

    # Activation checkpointing reduces activation memory dramatically
    if parallel.activation_checkpointing:
        act_gb *= 0.1  # ~10x reduction with per-layer checkpointing

    # Sequence parallelism further reduces per-GPU activation
    if parallel.sequence_parallel:
        act_gb /= tp  # Sequence dim split in SP-TP combined

    return MemoryEstimate(
        params_gb=params_gb,
        grads_gb=grads_gb,
        optimizer_gb=opt_gb,
        activations_gb=act_gb,
    )


# ---------------------------------------------------------------------------
# Communication overhead estimation
# ---------------------------------------------------------------------------


@dataclass
class CommunicationEstimate:
    """Estimated communication overhead for a parallel configuration."""

    intra_node_comm_gb_per_step: float = 0.0
    inter_node_comm_gb_per_step: float = 0.0
    total_comm_gb_per_step: float = 0.0
    intra_node_time_ms: float = 0.0
    inter_node_time_ms: float = 0.0
    total_comm_time_ms: float = 0.0

    def __post_init__(self) -> None:
        self.total_comm_gb_per_step = (
            self.intra_node_comm_gb_per_step + self.inter_node_comm_gb_per_step
        )


def estimate_communication(
    model: ModelSpec,
    parallel: ParallelConfig,
    batch_size: int,
    seq_len: int,
    hardware: HardwareConfig,
) -> CommunicationEstimate:
    """
    Estimate communication overhead for a parallel configuration.

    TP communication (intra-node NVLink):
    - 4 All-Reduces per transformer layer per step (2 fwd + 2 bwd)
    - All-Reduce volume = 2 * (tp-1)/tp * tensor_size per op
    - Tensor size = batch * seq * hidden * dtype_bytes

    PP communication (inter-stage, intra or inter-node):
    - 2 sends + 2 recvs per microbatch per stage boundary
    - Volume = batch_per_microbatch * seq * hidden * dtype_bytes * 2 (fwd+bwd)

    DP communication (All-Reduce gradients, intra or inter-node):
    - 1 All-Reduce per step for full model gradients
    - Volume = total_params * dtype_bytes

    ZeRO-3 FSDP communication:
    - All-Gather params: total_params * dtype_bytes (forward)
    - All-Gather + Reduce-Scatter: 2 * total_params * dtype_bytes (backward)
    - Total: 3 * total_params * dtype_bytes per step

    Args:
        model: Model specification.
        parallel: Parallel configuration.
        batch_size: Global batch size.
        seq_len: Sequence length.
        hardware: Hardware configuration.

    Returns:
        CommunicationEstimate with per-step communication breakdown.
    """
    total_params = count_parameters(model)
    dtype_size = model.dtype_bytes

    intra_comm = 0.0
    inter_comm = 0.0

    tp = parallel.tp_size
    pp = parallel.pp_size
    dp = parallel.dp_size

    # TP communication: 4 All-Reduces per layer per step
    # Each All-Reduce transfers 2 * (tp-1)/tp * data_size
    if tp > 1:
        tp_data_per_op = batch_size * seq_len * model.hidden_size * dtype_size
        tp_comm_per_op = 2 * (tp - 1) / tp * tp_data_per_op
        tp_comm_per_layer = 4 * tp_comm_per_op  # 4 All-Reduces per layer
        tp_total_comm = tp_comm_per_layer * model.num_layers
        intra_comm += tp_total_comm  # TP stays within NVLink domain

    # PP communication: send/recv between stages
    if pp > 1:
        pp_data_per_transfer = (
            (batch_size / dp) * seq_len * model.hidden_size * dtype_size
        )
        pp_transfers = 2 * parallel.num_microbatches * (pp - 1) * 2  # fwd+bwd
        pp_total_comm = pp_data_per_transfer * pp_transfers

        # PP may cross nodes; allocate proportionally
        nodes_used = max(1, pp / hardware.gpus_per_node)
        if nodes_used > 1:
            inter_ratio = (nodes_used - 1) / nodes_used
            inter_comm += pp_total_comm * inter_ratio
            intra_comm += pp_total_comm * (1 - inter_ratio)
        else:
            intra_comm += pp_total_comm

    # DP/ZeRO communication: gradient synchronization
    if dp > 1:
        if parallel.zero_stage == 0:
            # DDP: one All-Reduce for all parameters
            dp_comm_per_rank = 2 * (dp - 1) / dp * total_params * dtype_size
            # DDP can use either intra or inter-node bandwidth
            nodes_used = max(1, dp / hardware.gpus_per_node)
            if nodes_used > 1:
                inter_ratio = (nodes_used - 1) / nodes_used
                inter_comm += dp_comm_per_rank * inter_ratio
                intra_comm += dp_comm_per_rank * (1 - inter_ratio)
            else:
                intra_comm += dp_comm_per_rank

        elif parallel.zero_stage >= 3:
            # FSDP/ZeRO-3: 3 * total_params per step
            fsdp_comm = 3 * total_params * dtype_size
            nodes_used = max(1, dp / hardware.gpus_per_node)
            if nodes_used > 1:
                inter_ratio = (nodes_used - 1) / nodes_used
                inter_comm += fsdp_comm * inter_ratio
                intra_comm += fsdp_comm * (1 - inter_ratio)
            else:
                intra_comm += fsdp_comm
        else:
            # ZeRO-1/2: same as DDP but with reduce-scatter
            zeero_comm = 2 * (dp - 1) / dp * total_params * dtype_size
            nodes_used = max(1, dp / hardware.gpus_per_node)
            if nodes_used > 1:
                inter_ratio = (nodes_used - 1) / nodes_used
                inter_comm += zeero_comm * inter_ratio
                intra_comm += zeero_comm * (1 - inter_ratio)
            else:
                intra_comm += zeero_comm

    # Convert to GB and compute time
    intra_comm_gb = intra_comm / 1e9
    inter_comm_gb = inter_comm / 1e9

    intra_time = (
        (intra_comm_gb / hardware.intra_node_bandwidth_gb_s * 1000)
        if hardware.intra_node_bandwidth_gb_s > 0
        else 0
    )
    inter_time = (
        (inter_comm_gb / hardware.inter_node_bandwidth_gb_s * 1000)
        if hardware.inter_node_bandwidth_gb_s > 0
        else 0
    )

    return CommunicationEstimate(
        intra_node_comm_gb_per_step=intra_comm_gb,
        inter_node_comm_gb_per_step=inter_comm_gb,
        intra_node_time_ms=intra_time,
        inter_node_time_ms=inter_time,
        total_comm_time_ms=intra_time + inter_time,
    )


# ---------------------------------------------------------------------------
# MFU computation
# ---------------------------------------------------------------------------


def compute_mfu(
    model: ModelSpec,
    parallel: ParallelConfig,
    batch_size: int,
    seq_len: int,
    hardware: HardwareConfig,
    compute_time_ms: float,
) -> float:
    """
    Compute Model FLOPs Utilization (MFU).

    MFU = actual_FLOPs / (peak_FLOPs * time_seconds)

    For transformer models, actual FLOPs per step ≈ 6 * P * tokens_per_step
    (including forward and backward passes):
    - Forward: 2 * P per token
    - Backward: 4 * P per token (2x forward due to gradient computation)
    - Total: ~6 * P per token per step

    where P = total parameters (excluding embeddings for FLOP estimation).

    Args:
        model: Model specification.
        parallel: Parallel configuration.
        batch_size: Global batch size.
        seq_len: Sequence length.
        hardware: Hardware configuration.
        compute_time_ms: Actual measured compute time per step in ms.

    Returns:
        MFU as a fraction (e.g., 0.60 = 60%).
    """
    total_params = count_parameters(model)
    tokens_per_step = batch_size * seq_len

    # FLOPs per step (forward + backward)
    # Following PaLM paper: flops ≈ 6 * N * tokens + O(N * tokens * layers)
    # Refined: consider attention FLOPs separately
    flops_per_token_fwd = 2 * total_params  # Approximate forward FLOPs per token
    flops_per_token_total = 6 * total_params  # Forward + backward

    total_flops = flops_per_token_total * tokens_per_step

    # Peak FLOPs across all GPUs
    num_gpus = parallel.tp_size * parallel.pp_size * parallel.dp_size
    peak_flops_per_second = hardware.peak_flops_tflops * 1e12 * num_gpus

    # Time in seconds for computation only
    actual_flops_per_second = total_flops / (compute_time_ms / 1000.0)

    mfu = actual_flops_per_second / peak_flops_per_second
    return min(mfu, 1.0)


# ---------------------------------------------------------------------------
# Parallel Recommendation
# ---------------------------------------------------------------------------


@dataclass
class ParallelRecommendation:
    """Recommended parallel strategy with memory and performance estimates."""

    config: ParallelConfig
    memory: MemoryEstimate
    communication: CommunicationEstimate
    fits_in_memory: bool = True
    estimated_mfu: float = 0.0
    estimated_step_time_ms: float = 0.0
    reasoning: str = ""
    feasibility_score: float = 1.0  # 0-1, higher is better


# ---------------------------------------------------------------------------
# Strategy Planner
# ---------------------------------------------------------------------------


class ParallelPlanner:
    """
    Automated parallel strategy planner.

    Given a model spec and hardware config, explores the PTD-P space
    and returns the best configuration based on memory constraints,
    communication efficiency, and estimated MFU.

    Planning heuristics:
    1. TP limited to NVLink domain (≤ gpus_per_node, typically ≤ 8)
    2. PP benefits from ≤ num_nodes stages (one per node for IB crossing)
    3. DP uses all remaining GPUs
    4. ZeRO-3 enables DP scaling across many nodes
    5. SP (sequence parallel) + TP reduces activation memory
    6. Activation checkpointing trades compute for memory

    Decision flow:
        model_size < 1B    → DDP (simplest)
        1B < model < 10B   → ZeRO-2 or ZeRO-3
        10B < model < 100B → ZeRO-3 + TP (up to 8-way)
        model > 100B       → Full 3D: TP + PP + DP + ZeRO
    """

    def __init__(
        self,
        model: ModelSpec,
        hardware: HardwareConfig,
        batch_size: int,
        seq_len: int,
    ):
        self.model = model
        self.hardware = hardware
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.total_params = count_parameters(model)

    def explore_strategies(
        self,
        max_tp: int = 8,
        max_pp: Optional[int] = None,
        min_microbatches: int = 1,
        max_microbatches: int = 128,
    ) -> list[ParallelRecommendation]:
        """
        Explore feasible (tp, pp, dp) combinations.

        Args:
            max_tp: Maximum tensor parallelism size.
            max_pp: Maximum pipeline parallelism size (defaults to num_nodes).
            min_microbatches: Minimum number of microbatches for PP.
            max_microbatches: Maximum number of microbatches.

        Returns:
            List of feasible ParallelRecommendation objects.
        """
        if max_pp is None:
            max_pp = max(1, self.hardware.num_nodes)

        total_gpus = self.hardware.num_gpus
        recommendations: list[ParallelRecommendation] = []

        # Enumerate (tp, pp, dp) combinations
        for tp in [1, 2, 4, 8]:
            if tp > total_gpus or tp > max_tp or tp > self.hardware.gpus_per_node:
                continue

            for pp in [1, 2, 4, 8, 16, 32]:
                if pp > max_pp or pp > total_gpus:
                    continue
                if tp * pp > total_gpus:
                    continue
                if total_gpus % (tp * pp) != 0:
                    continue

                dp = total_gpus // (tp * pp)

                # Try different ZeRO stages
                for zero_stage in [0, 1, 2, 3]:
                    # Skip ZeRO configurations with no DP
                    if dp == 1 and zero_stage > 0:
                        continue

                    # Determine microbatches for PP
                    num_mb = 1
                    if pp > 1:
                        # Try to find a good microbatch count
                        num_mb = self._optimize_microbatches(
                            pp, min_microbatches, max_microbatches
                        )

                    config = ParallelConfig(
                        tp_size=tp,
                        pp_size=pp,
                        dp_size=dp,
                        zero_stage=zero_stage,
                        num_microbatches=num_mb,
                        sequence_parallel=(tp > 1),
                        activation_checkpointing=(pp > 1 or zero_stage >= 3),
                    )

                    mem = estimate_memory(
                        self.model, config, self.batch_size, self.seq_len, self.hardware
                    )
                    comm = estimate_communication(
                        self.model, config, self.batch_size, self.seq_len, self.hardware
                    )
                    fits = mem.total_gb <= self.hardware.gpu_memory_gb

                    if fits:
                        # Estimate compute and communication time
                        compute_time = self._estimate_compute_time(config)
                        total_step_time = compute_time + comm.total_comm_time_ms
                        mfu = compute_mfu(
                            self.model,
                            config,
                            self.batch_size,
                            self.seq_len,
                            self.hardware,
                            compute_time,
                        )

                        rec = ParallelRecommendation(
                            config=config,
                            memory=mem,
                            communication=comm,
                            fits_in_memory=True,
                            estimated_mfu=mfu,
                            estimated_step_time_ms=total_step_time,
                            reasoning=self._generate_reasoning(config, mem, comm),
                            feasibility_score=mfu
                            * (1.0 - mem.total_gb / self.hardware.gpu_memory_gb),
                        )
                        recommendations.append(rec)

        # Sort by feasibility score (higher MFU, lower memory pressure)
        recommendations.sort(key=lambda r: r.feasibility_score, reverse=True)
        return recommendations

    def recommend(self) -> ParallelRecommendation:
        """
        Return the best parallel strategy for the given configuration.

        If no configuration fits in memory, returns the one with the
        lowest memory usage (even if it's over capacity).

        Returns:
            Best ParallelRecommendation.
        """
        strategies = self.explore_strategies()

        if not strategies:
            # No config fits; return a default with maximum sharding
            return self._fallback_recommendation()

        fits = [r for r in strategies if r.fits_in_memory]
        if fits:
            # Pick the one with best MFU among those that fit
            fits.sort(key=lambda r: r.estimated_mfu, reverse=True)
            return fits[0]

        # Pick the one with lowest memory usage
        strategies.sort(key=lambda r: r.memory.total_gb)
        return strategies[0]

    def _optimize_microbatches(self, pp_size: int, min_mb: int, max_mb: int) -> int:
        """Find optimal number of microbatches for pipeline parallelism."""
        # Target bubble ratio ≤ 20%
        target_bubble = 0.2
        min_m = max(1, int(math.ceil((pp_size - 1) / target_bubble - pp_size + 1)))
        # Don't exceed what batch allows
        max_m_from_batch = self.batch_size
        m = min(min_m, max_m_from_batch, max_mb)
        m = max(m, min_mb)
        return m

    def _estimate_compute_time(self, config: ParallelConfig) -> float:
        """
        Estimate pure computation time per step (no communication).

        Uses roofline model: either compute-bound or memory-bound.
        For large transformer models, typically compute-bound on H100.

        Compute time ≈ 6 * params_per_gpu * tokens_per_gpu / flops_per_gpu
        """
        total_params = self.total_params
        num_gpus = config.tp_size * config.pp_size * config.dp_size
        params_per_gpu = total_params / (config.tp_size * config.pp_size)
        tokens_per_gpu = self.batch_size * self.seq_len / config.dp_size
        flops_per_gpu = self.hardware.peak_flops_tflops * 1e12

        # total flops ≈ 6 * params * tokens
        compute_flops = 6 * params_per_gpu * tokens_per_gpu

        # Time in ms
        compute_time = compute_flops / flops_per_gpu * 1000
        return max(compute_time, 0.01)

    def _generate_reasoning(
        self,
        config: ParallelConfig,
        mem: MemoryEstimate,
        comm: CommunicationEstimate,
    ) -> str:
        """Generate human-readable reasoning for this configuration."""
        parts = []
        parts.append(f"TP={config.tp_size} PP={config.pp_size} DP={config.dp_size}")
        if config.zero_stage > 0:
            parts.append(f"ZeRO-{config.zero_stage}")
        parts.append(
            f"Memory: {mem.total_gb:.1f}GB/{self.hardware.gpu_memory_gb:.0f}GB"
        )
        parts.append(f"Comm: {comm.total_comm_time_ms:.1f}ms/step")
        return "; ".join(parts)

    def _fallback_recommendation(self) -> ParallelRecommendation:
        """Generate a fallback recommendation using maximum sharding."""
        num_gpus = self.hardware.num_gpus
        config = ParallelConfig(
            tp_size=min(8, num_gpus),
            pp_size=1,
            dp_size=max(1, num_gpus // 8),
            zero_stage=3,
            num_microbatches=1,
            activation_checkpointing=True,
        )
        mem = estimate_memory(
            self.model, config, self.batch_size, self.seq_len, self.hardware
        )
        comm = estimate_communication(
            self.model, config, self.batch_size, self.seq_len, self.hardware
        )
        return ParallelRecommendation(
            config=config,
            memory=mem,
            communication=comm,
            fits_in_memory=mem.total_gb <= self.hardware.gpu_memory_gb,
            estimated_mfu=0.0,
            estimated_step_time_ms=comm.total_comm_time_ms + 100.0,
            reasoning="FALLBACK: Maximum sharding (ZeRO-3 + activation checkpointing). "
            "Consider using more GPUs or reducing batch size.",
            feasibility_score=0.1,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def plan_parallel_strategy(
    model: ModelSpec,
    hardware: Optional[HardwareConfig] = None,
    batch_size: int = 1024,
    seq_len: int = 4096,
    verbose: bool = False,
) -> ParallelRecommendation:
    """
    Plan the optimal parallel strategy for a given model and hardware.

    Args:
        model: Model specification.
        hardware: Hardware configuration (defaults to 8x H100).
        batch_size: Global batch size.
        seq_len: Sequence length.
        verbose: Print detailed analysis.

    Returns:
        Best ParallelRecommendation.
    """
    if hardware is None:
        hardware = HardwareConfig.h100_8gpu()

    planner = ParallelPlanner(model, hardware, batch_size, seq_len)

    if verbose:
        total_params = count_parameters(model)
        print(f"\n{'=' * 70}")
        print(f"Parallel Strategy Planner")
        print(f"{'=' * 70}")
        print(f"Model: {model.name}")
        print(f"Parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
        print(
            f"Hardware: {hardware.num_gpus}x {hardware.gpu_model} "
            f"({hardware.gpu_memory_gb:.0f}GB each)"
        )
        print(f"Batch: {batch_size}, Seq: {seq_len}")
        print()

    recommendation = planner.recommend()

    if verbose:
        cfg = recommendation.config
        mem = recommendation.memory
        comm = recommendation.communication
        print(
            f"\nRecommended: TP={cfg.tp_size} PP={cfg.pp_size} DP={cfg.dp_size} "
            f"ZeRO-{cfg.zero_stage} microbatches={cfg.num_microbatches}"
        )
        print(
            f"Memory: {mem.total_gb:.1f}GB ({mem.params_gb:.1f}P + "
            f"{mem.grads_gb:.1f}G + {mem.optimizer_gb:.1f}O + {mem.activations_gb:.1f}A)"
        )
        print(f"Fits: {'YES' if recommendation.fits_in_memory else 'NO'}")
        print(
            f"Communication: {comm.total_comm_time_ms:.1f}ms/step "
            f"({comm.intra_node_comm_gb_per_step:.2f}GB intra, "
            f"{comm.inter_node_comm_gb_per_step:.2f}GB inter)"
        )
        print(f"Est. MFU: {recommendation.estimated_mfu:.1%}")
        print(f"\n{recommendation.reasoning}")

    return recommendation
