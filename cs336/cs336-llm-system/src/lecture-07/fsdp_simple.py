"""
Explain FSDP (Fully Sharded Data Parallel) / ZeRO stages with code comments.
No actual distributed execution; this file documents the memory-saving strategy
at each ZeRO stage with annotated pseudo-code.
"""

from __future__ import annotations

import torch


# =====================================================================
# ZeRO Stage Overview
# =====================================================================
#
# Training a model requires storing:
#   1. Model parameters (P)
#   2. Gradients       (G)   - same size as parameters
#   3. Optimizer states (O)  - e.g., momentum + variance in Adam = 2P
#
# Without parallelism, each GPU stores: P + G + O = P + P + 2P = 4P (for Adam)
#
# Data Parallel (DP): each GPU has full P, G, O → 4P per GPU
# DDP: each GPU has full P, G, O → 4P per GPU (same memory, comm different)
#
# ZeRO-1 (Optimizer State Sharding):
#   - Partition O across GPUs: each stores O/N instead of O
#   - Memory: P + G + O/N = P + P + 2P/N
#   - Savings: Optimizer states are the largest consumer in Adam
#
# ZeRO-2 (Gradient Sharding):
#   - Additionally partition G across GPUs
#   - Memory: P + G/N + O/N = P + P/N + 2P/N
#   - After backward, gradients are reduce-scattered (not all-reduced)
#
# ZeRO-3 (Parameter Sharding):
#   - Additionally partition P across GPUs
#   - Memory: P/N + G/N + O/N = (P + P + 2P) / N = 4P/N
#   - All-gather parameters before each layer's forward/backward
#   - Discard parameters after layer is done
#
# FSDP in PyTorch implements ZeRO-3 semantics.
# =====================================================================


def print_zeero_stages() -> None:
    """Print memory comparison across ZeRO stages."""
    print("=" * 70)
    print("ZeRO Stage Memory Analysis (for Adam optimizer)")
    print("=" * 70)

    # Example: 1B parameter model, 4 GPUs
    P = 1e9  # number of parameters
    N = 4  # number of GPUs
    bytes_per_param = 4  # fp32

    param_mem = P * bytes_per_param / 1e9  # GB
    grad_mem = P * bytes_per_param / 1e9  # GB
    opt_mem = 2 * P * bytes_per_param / 1e9  # GB (Adam: m + v)

    print(f"\nModel: {P / 1e9:.1f}B parameters, {N} GPUs, fp32")
    print(f"  Parameter memory:  {param_mem:.1f} GB")
    print(f"  Gradient memory:   {grad_mem:.1f} GB")
    print(f"  Optimizer memory:  {opt_mem:.1f} GB (Adam: momentum + variance)")
    print()

    strategies = {
        "Naive DP / DDP": (1.0, 1.0, 1.0),
        "ZeRO-1 (OS)   ": (1.0, 1.0, 1.0 / N),
        "ZeRO-2 (OS+G) ": (1.0, 1.0 / N, 1.0 / N),
        "ZeRO-3 (OS+G+P)": (1.0 / N, 1.0 / N, 1.0 / N),
    }

    print(
        f"{'Strategy':<18} {'Params (GB)':<12} {'Grads (GB)':<12} {'Opt (GB)':<12} {'Total (GB)':<12}"
    )
    print("-" * 66)
    for name, (pf, gf, of_) in strategies.items():
        p_mem = param_mem * pf
        g_mem = grad_mem * gf
        o_mem = opt_mem * of_
        total = p_mem + g_mem + o_mem
        print(
            f"  {name:<16} {p_mem:<12.2f} {g_mem:<12.2f} {o_mem:<12.2f} {total:<12.2f}"
        )


def print_zeero3_workflow() -> None:
    """Explain ZeRO-3 (FSDP) forward/backward workflow with comments."""
    print("\n" + "=" * 70)
    print("ZeRO-3 / FSDP Workflow (per-layer execution)")
    print("=" * 70)
    print("""
    For each transformer block i in the model:

    1. All-gather parameters for block i:
       - Root process gathers sharded parameters from all GPUs
       - Reconstructs the full weight tensor for block i
       - Communication: all-gather (size = full_block_params)

    2. Forward pass through block i:
       - Compute with full parameters
       - Discard the gathered parameters (free memory)
       - Keep activations for backward

    3. (After all blocks done) Compute loss and start backward:

    4. For each block i in reverse order:
       a. All-gather parameters for block i again
       b. Compute backward pass through block i
       c. Reduce-scatter gradients for block i
          (each GPU keeps only its shard of the gradient)
       d. Discard full parameters

    5. Update sharded parameters with sharded gradients (optimizer step):
       - Each GPU only updates its own parameter shard
       - Optimizer states also sharded → per-GPU memory is O/N

    Key insight: Parameters are "materialized" (fully gathered) only
    one layer at a time. At any given moment, only one layer's full
    parameters reside in memory, drastically reducing peak memory.
    """)

    # Diagram as text
    print("Memory timeline for a 3-layer model over 4 GPUs:")
    print()
    print(
        "  Time →   [Forward Layer1] [Forward Layer2] [Forward Layer3]  [Loss]   [Bwd L3] [Bwd L2] [Bwd L1]"
    )
    print(
        "  GPU 0:   [L1_full+L1_shard] [L2_full+L1_act] [L3_full+L1,L2_act] [acts]  [L3_full] [L2_full] [L1_full]"
    )
    print(
        "  GPU 1:   [L1_full+L2_shard] [L2_full+        [L3_full+           [acts]  [         [         [         "
    )
    print(
        "  GPU 2:   [L1_full+L3_shard] [L2_full+        [L3_full+           [acts]  [         [         [         "
    )
    print(
        "  GPU 3:   [L1_full+L4_shard] [L2_full+        [L3_full+           [acts]  [         [         [         "
    )
    print()
    print("  Each full layer is temporarily materialized via all-gather, then freed.")
    print("  Each GPU permanently stores its parameter shard + optimizer shard.")


def print_communication_patterns() -> None:
    """Show the communication patterns used by each ZeRO stage."""
    print("\n" + "=" * 70)
    print("Communication Patterns per Step")
    print("=" * 70)
    print("""
    ZeRO-1:
      - Forward:  None (parameters already local)
      - Backward: Reduce-scatter gradients (same volume as all-reduce)
      - Optimizer: Each GPU updates its own optimizer partition

    ZeRO-2:
      - Forward:  None
      - Backward: Reduce-scatter gradients (same volume as ZeRO-1)
      - Optimizer: Each GPU updates its own optimizer partition
      - Note: Communication volume same as DDP, just rearranged

    ZeRO-3:
      - Forward:  All-gather parameters (one layer at a time)
                   → Adds communication: P * bytes_per_layer per layer
                   → Total: P * bytes_per_param sent per step
      - Backward: All-gather parameters + reduce-scatter gradients
                   → Total: 2P * bytes_per_param sent per step
                   → 1.5x the communication of DDP

    Trade-off:  More communication for less memory.
               ZeRO-3:  1.5x comm, 1/N memory
               DDP:     1.0x comm, 1.0x memory
    """)


def main() -> None:
    print_zeero_stages()
    print_zeero3_workflow()
    print_communication_patterns()


if __name__ == "__main__":
    main()
