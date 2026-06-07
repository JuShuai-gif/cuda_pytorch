#!/usr/bin/env python3
"""
MIT 6.5940 Lecture 20: Gradient Compression and Hybrid Parallelism

Topics covered:
  - Simulate Deep Gradient Compression: top-k sparsification + momentum
    correction
  - 1-Bit SGD simulation: quantize gradients to 1-bit with error feedback
  - Compare: communication volume reduction vs accuracy impact
  - Hybrid parallelism memory calculation: DP + PP + TP on a given model

All computation runs on CPU.  No GPU required.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Seed
# ===========================================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ===========================================================================
# 1. Deep Gradient Compression (DGC): Top-k Sparsification
# ===========================================================================


class DeepGradientCompression:
    """Implements DGC with momentum correction and error feedback.

    Reference: Lin et al., "Deep Gradient Compression: Reducing the
    Communication Bandwidth for Distributed Training", ICLR 2018.

    Algorithm:
      1. Select top-k% largest gradients by magnitude (sparsification)
      2. Accumulate small gradients in error residual (error feedback)
      3. Apply momentum correction to compensate for staleness
    """

    def __init__(self, sparsity: float = 0.99, momentum: float = 0.9):
        """
        Args:
            sparsity: fraction of gradients to zero out (0.99 = keep top 1%)
            momentum: momentum factor for correction
        """
        self.sparsity = sparsity
        self.momentum = momentum
        self.residual: torch.Tensor | None = None
        self.momentum_buffer: torch.Tensor | None = None

    def compress(self, grad: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Compress gradient tensor using top-k sparsification.

        Args:
            grad: gradient tensor to compress

        Returns:
            (compressed gradient, number of non-zero elements)
        """
        # Add residual (error feedback)
        if self.residual is not None and self.residual.shape == grad.shape:
            grad = grad + self.residual

        g = grad.flatten()
        total_elements = g.numel()
        k = max(1, int(total_elements * (1.0 - self.sparsity)))

        # Top-k: keep only the k largest absolute values
        _, indices = torch.topk(g.abs(), k)
        mask = torch.zeros_like(g)
        mask[indices] = 1.0
        compressed = g * mask

        # Store residual (the values not sent)
        self.residual = (g - compressed).reshape(grad.shape)

        # Momentum correction
        if self.momentum_buffer is None or self.momentum_buffer.shape != grad.shape:
            self.momentum_buffer = torch.zeros_like(grad)
        compressed_reshaped = compressed.reshape(grad.shape)
        self.momentum_buffer = (
            self.momentum * self.momentum_buffer + compressed_reshaped
        )

        return self.momentum_buffer.clone(), k


def simulate_dgc_impact(
    grad_magnitudes: List[float], sparsities: List[float]
) -> Dict[float, Dict[str, float]]:
    """Simulate the impact of DGC on gradient reconstruction quality.

    Args:
        grad_magnitudes: sorted list of gradient magnitudes (descending)
        sparsities: list of sparsity ratios to test

    Returns:
        Dictionary mapping sparsity -> {energy_retained, compression_ratio}
    """
    results = {}
    g = torch.tensor(grad_magnitudes, dtype=torch.float32)
    total_energy = (g**2).sum().item()

    for s in sparsities:
        k = max(1, int(len(g) * (1.0 - s)))
        topk_energy = (g[:k] ** 2).sum().item()
        energy_retained = topk_energy / total_energy if total_energy > 0 else 1.0
        compression = 1.0 / (1.0 - s) if s < 1.0 else float("inf")
        results[s] = {
            "energy_retained": energy_retained,
            "compression_ratio": compression,
            "values_sent": k,
        }
    return results


# ===========================================================================
# 2. 1-Bit SGD with Error Feedback
# ===========================================================================


class OneBitSGD:
    """1-Bit SGD: quantize gradients to 1 bit with error feedback.

    Reference: Seide et al., "1-Bit Stochastic Gradient Descent and its
    Application to Data-Parallel Distributed Training of Speech DNNs",
    Interspeech 2014.

    Algorithm:
      1. Add residual error to current gradient
      2. Compute sign(grad) * mean(|grad|) — 1 bit for sign, 1 scalar for scale
      3. Update residual with quantization error
    """

    def __init__(self):
        self.residual: torch.Tensor | None = None

    def compress(self, grad: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Compress gradient to 1-bit representation.

        Args:
            grad: gradient tensor

        Returns:
            (quantized gradient, effective bits communicated)
        """
        # Add error feedback
        if self.residual is not None and self.residual.shape == grad.shape:
            grad = grad + self.residual

        # 1-bit quantization: sign only
        scale = grad.abs().mean()
        quantized = torch.sign(grad) * scale

        # Store error
        self.residual = grad - quantized

        # Communication: 1 bit per element + 1 float32 scalar
        bits = grad.numel() * 1 + 32
        return quantized, bits


def compare_compression_methods(
    grad: torch.Tensor,
    sparsity: float = 0.99,
) -> Dict[str, Dict[str, float]]:
    """Compare DGC and 1-Bit SGD on the same gradient tensor.

    Args:
        grad: sample gradient tensor
        sparsity: DGC sparsity ratio

    Returns:
        Comparison metrics for each method.
    """
    orig_size = grad.numel() * 32  # bits (FP32)

    # DGC
    dgc = DeepGradientCompression(sparsity=sparsity)
    dgc_comp, dgc_nnz = dgc.compress(grad)
    dgc_bits = dgc_nnz * 32 + dgc_nnz * math.ceil(
        math.log2(grad.numel())
    )  # value + index
    dgc_cosine = F.cosine_similarity(grad.flatten(), dgc_comp.flatten(), dim=0).item()

    # 1-Bit SGD
    onebit = OneBitSGD()
    onebit_comp, onebit_bits = onebit.compress(grad)
    onebit_cosine = F.cosine_similarity(
        grad.flatten(), onebit_comp.flatten(), dim=0
    ).item()

    return {
        "original": {"bits": float(orig_size), "compression": 1.0},
        "dgc": {
            "bits": float(dgc_bits),
            "compression": orig_size / max(dgc_bits, 1),
            "cosine_sim": dgc_cosine,
        },
        "1bit_sgd": {
            "bits": float(onebit_bits),
            "compression": orig_size / max(onebit_bits, 1),
            "cosine_sim": onebit_cosine,
        },
    }


# ===========================================================================
# 3. Hybrid Parallelism Memory Calculator
# ===========================================================================


def hybrid_parallelism_memory(
    model_params: int,
    hidden_dim: int,
    num_layers: int,
    dp_size: int = 1,
    pp_size: int = 1,
    tp_size: int = 1,
    batch_size: int = 64,
    seq_len: int = 512,
) -> Dict[str, float]:
    """Calculate memory usage for hybrid DP + PP + TP.

    Memory components:
      - Model parameters (P): partitioned by PP (per stage) and TP
      - Gradients (G): same partition as parameters
      - Optimizer states (O): Adam m+v, partitioned same as P
      - Activations (A): proportional to batch/seq per PP stage

    Args:
        model_params: total parameter count
        hidden_dim: model hidden dimension
        num_layers: number of transformer layers
        dp_size: data-parallel replicas
        pp_size: pipeline-parallel stages
        tp_size: tensor-parallel size
        batch_size: global batch size
        seq_len: sequence length

    Returns:
        Memory breakdown per device in GB.
    """
    total_devices = dp_size * pp_size * tp_size
    device_factor = pp_size * tp_size

    # Parameters: partitioned across PP and TP, replicated across DP
    P = model_params * 4 / device_factor

    # Gradients: same partitioning
    G = P

    # Optimizer states (Adam): m + v
    O = P * 2

    # Activations (rough estimate): 34 * b * s * h per transformer layer
    # per PP micro-batch, stored for backward pass
    micro_batch = max(1, batch_size // dp_size)
    layers_per_stage = max(1, num_layers // pp_size)
    A = 34 * micro_batch * seq_len * hidden_dim * layers_per_stage * 4

    total_mem = P + G + O + A

    return {
        "config": f"DP={dp_size} PP={pp_size} TP={tp_size} (total_devices={total_devices})",
        "params_gb": P / 1e9,
        "grads_gb": G / 1e9,
        "optimizer_gb": O / 1e9,
        "activations_gb": A / 1e9,
        "total_mem_gb": total_mem / 1e9,
    }


# ===========================================================================
# 4. Gradient distributions for DGC simulation
# ===========================================================================


def generate_power_law_gradients(n: int, alpha: float = 1.5) -> torch.Tensor:
    """Generate gradient magnitudes following power-law distribution.

    Real neural network gradients often follow a power-law, making
    top-k sparsification highly effective.
    """
    ranks = np.arange(1, n + 1)
    magnitudes = ranks ** (-alpha)
    magnitudes = magnitudes / magnitudes.sum()
    # Add noise
    noise = np.random.normal(0, 0.01, n)
    magnitudes = np.abs(magnitudes + noise)
    magnitudes.sort()
    return torch.tensor(magnitudes[::-1].copy(), dtype=torch.float32)


# ===========================================================================
# 5. Main demonstration
# ===========================================================================


def main() -> None:
    print("=" * 72)
    print("MIT 6.5940 Lecture 20: Gradient Compression & Hybrid Parallelism")
    print("=" * 72)

    # ---------- DGC: Top-k Sparsification ----------
    print("\n--- 1. Deep Gradient Compression (DGC) ---")
    n_elements = 1_000_000
    grad = generate_power_law_gradients(n_elements, alpha=1.5)

    print(f"  Gradient tensor size: {n_elements:,} elements")
    print(f"  Distribution: power-law (alpha=1.5)")

    sparsities = [0.90, 0.95, 0.99, 0.999]
    dgc_results = simulate_dgc_impact(grad.tolist(), sparsities)
    print(
        f"  {'Sparsity':>10} {'Energy Retained':>16} {'Compression':>14} {'Values Sent':>13}"
    )
    print(f"  {'-' * 55}")
    for s in sparsities:
        r = dgc_results[s]
        print(
            f"  {s:>10.1%} {r['energy_retained']:>16.1%} {r['compression_ratio']:>14.1f}x "
            f"{r['values_sent']:>13,}"
        )

    # ---------- DGC with momentum correction ----------
    print("\n--- 2. DGC with Momentum Correction ---")
    dgc = DeepGradientCompression(sparsity=0.99, momentum=0.9)
    sample_grad = torch.randn(100, 100)
    compressed, nnz = dgc.compress(sample_grad)
    orig_norm = sample_grad.norm().item()
    comp_norm = compressed.norm().item()
    print(f"  Original grad norm: {orig_norm:.4f}")
    print(f"  Compressed grad norm: {comp_norm:.4f}")
    print(f"  Non-zero ratio: {nnz / sample_grad.numel():.2%}")

    # Simulate multiple steps with error feedback
    print("  Multi-step error feedback simulation:")
    dgc2 = DeepGradientCompression(sparsity=0.99, momentum=0.9)
    grad_sequence = [torch.randn(100) * 0.1 + torch.ones(100) for _ in range(5)]
    for step, g in enumerate(grad_sequence):
        compressed, nnz = dgc2.compress(g)
        error_norm = dgc2.residual.norm().item() if dgc2.residual is not None else 0
        print(f"    Step {step}: nnz={nnz}/{g.numel()}, residual norm={error_norm:.4f}")

    # ---------- 1-Bit SGD ----------
    print("\n--- 3. 1-Bit SGD Simulation ---")
    sample_grad_1bit = torch.randn(10000)
    onebit = OneBitSGD()

    for step in range(3):
        grad_step = sample_grad_1bit + torch.randn(10000) * 0.01
        quantized, bits = onebit.compress(grad_step)
        # Simulate: only sign matters for direction
        sign_match = (
            (torch.sign(grad_step) == torch.sign(quantized)).float().mean().item()
        )
        print(
            f"  Step {step}: bits={bits} ({bits / (grad_step.numel() * 32):.1%} of FP32), "
            f"sign_match={sign_match:.1%}"
        )

    # ---------- Compression Comparison ----------
    print("\n--- 4. Compression Methods Comparison ---")
    test_grad = generate_power_law_gradients(10000, alpha=1.2)
    comparison = compare_compression_methods(test_grad, sparsity=0.99)
    print(f"  {'Method':<14} {'Bits':>10} {'Compression':>14} {'Cosine Sim':>12}")
    print(f"  {'-' * 52}")
    for method, metrics in comparison.items():
        cs = (
            f"{metrics.get('cosine_sim', 1.0):.4f}"
            if "cosine_sim" in metrics
            else "1.0000"
        )
        print(
            f"  {method:<14} {metrics['bits']:>10.0f} {metrics['compression']:>14.1f}x {cs:>12}"
        )

    # ---------- Hybrid Parallelism Memory ----------
    print("\n--- 5. Hybrid Parallelism (DP+PP+TP) Memory ---")
    # Simulate a GPT-3-like model (175B params, 96 layers, hidden=12288)
    model_params_large = 175_000_000_000
    hidden_dim_large = 12288
    num_layers_large = 96

    configs = [
        (64, 1, 1),  # Pure DP: 64-way data parallel
        (8, 8, 1),  # DP+PP: moderate
        (4, 4, 4),  # DP+PP+TP: balanced
        (1, 16, 4),  # PP-heavy + TP
        (1, 8, 8),  # TP-heavy
    ]

    print(f"  Model: ~175B params, 96 layers, hidden={hidden_dim_large}")
    print(
        f"  {'Config':<30} {'Params':>8} {'Grads':>8} {'Opt':>8} "
        f"{'Acts':>8} {'Total':>8}"
    )
    print(f"  {'-' * 72}")

    for dp, pp, tp in configs:
        mem = hybrid_parallelism_memory(
            model_params_large,
            hidden_dim_large,
            num_layers_large,
            dp,
            pp,
            tp,
            batch_size=2048,
            seq_len=2048,
        )
        print(
            f"  {mem['config']:<30} {mem['params_gb']:>8.2f} {mem['grads_gb']:>8.2f} "
            f"{mem['optimizer_gb']:>8.2f} {mem['activations_gb']:>8.2f} "
            f"{mem['total_mem_gb']:>8.2f}"
        )

    # ---------- Communication Volume Analysis ----------
    print("\n--- 6. Communication Volume Reduction ---")
    grad_size_gb = 175e9 * 4 / 1e9  # ~700 GB
    print(f"  Baseline gradient size: {grad_size_gb:.1f} GB (FP32)")

    for s, label in [(0.99, "DGC (1% remaining)"), (0.999, "DGC (0.1% remaining)")]:
        comp_vol = grad_size_gb * (1 - s)
        print(f"  {label:<22}: {comp_vol:.1f} GB  ({1 / (1 - s):.0f}x compression)")

    onebit_vol = grad_size_gb / 32  # 1 bit per value vs 32 bits
    print(f"  1-Bit SGD {'':<14}: {onebit_vol:.1f} GB  (32x compression)")

    # ---------- Summary ----------
    print("\n--- 7. Summary ---")
    print("  Key takeaways:")
    print("    - DGC: 100-1000x gradient compression with <1% accuracy loss")
    print("    - 1-Bit SGD: 32x compression, works well with error feedback")
    print("    - Error feedback is critical: residuals prevent information loss")
    print("    - Momentum correction compensates for staleness in DGC")
    print("    - Hybrid parallelism (DP+PP+TP) essential for 100B+ models")
    print("    - Communication is the bottleneck in distributed training")

    print("\nDone. All computations on CPU.\n")


if __name__ == "__main__":
    main()
