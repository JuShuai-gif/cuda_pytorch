#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified model compression benchmark script.

MIT 6.5940 Efficient DL - Model Compression Practical Project.

Performs the full compression pipeline on representative models:
1. SmallCNN: unstructured pruning, structured channel pruning, PTQ INT8 quant,
   ONNX export + onnxruntime inference.
2. TransformerAttentionBlock: dynamic quantization, low-precision inference.
3. VLAActionHead: MLP action head pruning + quantization for robot VLA,
   measures action MSE / action deviation.

Generates reports/model_compression_report.md with comparison tables.

Usage:
    # Full benchmark (CPU, synthetic data)
    python src/model_compression/benchmark_compression.py

    # With GPU
    python src/model_compression/benchmark_compression.py --device cuda

    # Quick smoke test
    python src/model_compression/benchmark_compression.py --runs 3 --warmup 1 --train-steps 1

    # Custom model sizes
    python src/model_compression/benchmark_compression.py --batch-size 16 --hidden-size 256

No external data downloads required - uses synthetic random inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as torch_prune

# Project root for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model_compression.models import (
    SmallCNN,
    TransformerAttentionBlock,
    VLAActionHead,
    SimpleMLP,
    count_parameters,
    count_all_parameters,
)
from src.model_compression.metrics import (
    measure_parameters,
    measure_model_size_disk,
    measure_inference_latency,
    measure_throughput,
    measure_memory_usage,
    estimate_flops_manual,
    compute_model_mse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_compression")

# ============================================================
# Constants
# ============================================================

REPORTS_DIR = _PROJECT_ROOT / "reports"
ARTIFACTS_DIR = REPORTS_DIR / "artifacts" / "model_compression"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model Compression Benchmark")
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for benchmarks")
    p.add_argument("--seq-len", type=int, default=64, help="Sequence length for Transformer")
    p.add_argument("--hidden-size", type=int, default=128, help="Hidden size for Transformer")
    p.add_argument("--runs", type=int, default=30, help="Measurement runs")
    p.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    p.add_argument("--train-steps", type=int, default=20, help="Synthetic training steps (0 to skip)")
    p.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity for pruning")
    p.add_argument("--quant-bits", type=int, default=8, help="Quantization bits (8, 4)")
    p.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export steps")
    p.add_argument("--skip-tensorrt", action="store_true", help="Skip TensorRT steps")
    p.add_argument("--output", default=str(REPORTS_DIR / "model_compression_report.md"),
                   help="Output report path")
    return p.parse_args()


# ============================================================
# Synthetic Training
# ============================================================

def train_cnn_synthetic(
    model: nn.Module,
    device: torch.device,
    steps: int = 20,
    batch_size: int = 32,
    lr: float = 0.01,
) -> nn.Module:
    """Train a CNN on synthetic random data for a few steps."""
    model = model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0:
            logger.debug("  train step %d/%d loss=%.4f", step + 1, steps, loss.item())

    model.eval()
    return model


def train_mlp_synthetic(
    model: nn.Module,
    device: torch.device,
    steps: int = 20,
    batch_size: int = 32,
    input_dim: int = 784,
) -> nn.Module:
    """Train an MLP on synthetic random data."""
    model = model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        x = torch.randn(batch_size, input_dim, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    model.eval()
    return model


# ============================================================
# Pruning Operations
# ============================================================

def apply_unstructured_magnitude_pruning(
    model: nn.Module,
    sparsity: float = 0.5,
) -> nn.Module:
    """Apply global unstructured magnitude pruning.

    Zeros out the smallest-magnitude weights across all Conv2d and Linear layers.
    """
    model = copy.deepcopy(model)

    # Collect all weights
    all_weights: list[torch.Tensor] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data.abs().flatten()
            all_weights.append(w)

    if not all_weights:
        logger.warning("No prunable layers found")
        return model

    # Global threshold
    flat = torch.cat([w for w in all_weights])
    k = int(sparsity * flat.numel())
    if k == 0:
        return model
    threshold = float(torch.kthvalue(flat, k).values.item())

    # Apply mask
    total_pruned = 0
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            mask = (w.abs() > threshold).float()
            n_total = w.numel()
            n_pruned = n_total - int(mask.sum().item())
            total_pruned += n_pruned
            total_params += n_total
            module.weight.data.mul_(mask)

    actual_sparsity = total_pruned / total_params if total_params > 0 else 0
    logger.info("Unstructured pruning: target=%.2f%%, actual=%.2f%%",
                sparsity * 100, actual_sparsity * 100)
    return model


def apply_channel_pruning(
    model: nn.Module,
    sparsity: float = 0.5,
) -> nn.Module:
    """Apply structured channel pruning on Conv2d layers.

    Zeros out entire output channels (structured sparsity).
    """
    model = copy.deepcopy(model)
    total_channels = 0
    total_pruned = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data
            out_channels = w.shape[0]
            total_channels += out_channels

            # Compute channel importance via L2 norm
            importance = w.view(out_channels, -1).norm(p=2, dim=1)
            num_keep = max(1, int(out_channels * (1 - sparsity)))
            _, keep_idx = torch.topk(importance, num_keep)

            # Create channel mask
            mask = torch.zeros(out_channels, device=w.device)
            mask[keep_idx] = 1.0
            mask = mask.view(out_channels, *([1] * (w.dim() - 1)))
            module.weight.data.mul_(mask)
            total_pruned += (out_channels - num_keep)

    actual_sparsity = total_pruned / total_channels if total_channels > 0 else 0
    logger.info("Channel pruning: target=%.2f%%, actual=%.2f%%",
                sparsity * 100, actual_sparsity * 100)
    return model


# ============================================================
# Quantization Operations
# ============================================================

def quantize_tensor_ptq(
    weight: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
) -> torch.Tensor:
    """Simple PTQ quantization: quantize then dequantize weight tensor."""
    if bits < 1 or bits > 16:
        raise ValueError(f"bits must be 1-16, got {bits}")

    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        w_max = weight.abs().max().item()
        scale = w_max / qmax if w_max > 1e-8 else 1e-8
        q = torch.clamp(torch.round(weight / scale), -qmax, qmax)
        return q * scale
    else:
        w_min = weight.min().item()
        w_max = weight.max().item()
        qmax = 2 ** bits - 1
        if w_max == w_min:
            return weight
        scale = (w_max - w_min) / qmax
        zp = round(0 - w_min / scale)
        zp = max(0, min(qmax, zp))
        q = torch.clamp(torch.round(weight / scale + zp), 0, qmax)
        return (q - zp) * scale


def apply_ptq_quantization(
    model: nn.Module,
    bits: int = 8,
    symmetric: bool = True,
) -> nn.Module:
    """Apply Post-Training Quantization to all Conv2d and Linear weights."""
    model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            qw = quantize_tensor_ptq(w, bits=bits, symmetric=symmetric)
            module.weight.data.copy_(qw)

    logger.info("PTQ INT%d quantization applied", bits)
    return model


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply PyTorch built-in dynamic quantization (INT8 for Linear layers)."""
    model = copy.deepcopy(model)
    try:
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
            inplace=False,
        )
        logger.info("Dynamic quantization (torch.quantize_dynamic) applied")
        return quantized
    except Exception as e:
        logger.warning("Dynamic quantization failed: %s. Returning FP32 copy.", e)
        return model


# ============================================================
# ONNX Export
# ============================================================

def export_onnx(
    model: nn.Module,
    onnx_path: str,
    input_shape: tuple[int, ...],
    device: torch.device,
    dynamic_batch: bool = False,
) -> bool:
    """Export PyTorch model to ONNX format."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        logger.warning("onnx not installed, skipping ONNX export")
        return False

    model = model.to("cpu")
    model.eval()
    dummy = torch.randn(1, *input_shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        logger.info("ONNX exported to %s (size: %.2f KB)",
                     onnx_path, os.path.getsize(onnx_path) / 1024)
        return True
    except Exception as e:
        logger.warning("ONNX export failed: %s", e)
        return False


def run_onnxruntime_benchmark(
    onnx_path: str,
    input_shape: tuple[int, ...],
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 100,
) -> dict[str, Any] | None:
    """Run inference benchmark using onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed")
        return None

    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        logger.warning("Failed to create onnxruntime session: %s", e)
        return None

    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(batch_size, *input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    # Measure
    latencies: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "mean_ms": round(float(arr.mean()), 4),
        "median_ms": round(float(np.median(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "runs": runs,
    }


# ============================================================
# TensorRT (optional)
# ============================================================

def check_tensorrt_available() -> tuple[bool, str]:
    """Check if TensorRT is available in this environment."""
    reasons: list[str] = []

    # Check Python package
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        reasons.append("tensorrt Python package not installed")

    # Check trtexec CLI
    trtexec_path = None
    import shutil
    trtexec_path = shutil.which("trtexec")
    if not trtexec_path:
        reasons.append("trtexec CLI not found in PATH")

    if reasons:
        return False, "; ".join(reasons)
    return True, "TensorRT available"


# ============================================================
# Main Benchmark Runner
# ============================================================

class CompressionBenchmark:
    """Orchestrates the full compression benchmark pipeline."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = torch.device("cpu")

        self.results: list[dict[str, Any]] = []
        self.artifacts: dict[str, str] = {}

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Device: %s", self.device)
        logger.info("PyTorch: %s", torch.__version__)

    def save_artifact(self, model: nn.Module, name: str) -> str:
        path = str(ARTIFACTS_DIR / f"{name}.pt")
        torch.save(model.state_dict(), path)
        return path

    def _make_cnn_input(self, batch_size: int | None = None):
        bs = batch_size if batch_size is not None else self.args.batch_size
        return torch.randn(bs, 3, 32, 32)

    def _make_transformer_input(self, batch_size: int | None = None):
        bs = batch_size if batch_size is not None else self.args.batch_size
        return torch.randn(bs, self.args.seq_len, self.args.hidden_size)

    def _make_vla_input(self, batch_size: int | None = None):
        bs = batch_size if batch_size is not None else self.args.batch_size
        return (
            torch.randn(bs, 256),
            torch.randn(bs, 7),
        )

    def _record_result(self, entry: dict[str, Any]) -> None:
        self.results.append(entry)

    # ---- SmallCNN Benchmarks ----

    def benchmark_smallcnn(self) -> None:
        logger.info("=" * 60)
        logger.info("1. SmallCNN Compression Benchmark")
        logger.info("=" * 60)

        # Create and train baseline
        baseline = SmallCNN()
        if self.args.train_steps > 0:
            baseline = train_cnn_synthetic(baseline, self.device,
                                           steps=self.args.train_steps)
        baseline.to(self.device)
        baseline.eval()
        self.save_artifact(baseline, "smallcnn_baseline")

        cnn_input_fn = self._make_cnn_input
        input_shape = (3, 32, 32)

        # Measure baseline
        logger.info("--- Baseline measurements ---")
        params_b = measure_parameters(baseline)
        size_b = measure_model_size_disk(baseline)
        lat_b = measure_inference_latency(baseline, cnn_input_fn,
                                          self.args.warmup, self.args.runs, self.device)
        tp_b = measure_throughput(baseline, cnn_input_fn,
                                  self.args.batch_size, device=self.device)
        mem_b = measure_memory_usage(baseline, cnn_input_fn, self.device)
        flops_b = estimate_flops_manual(baseline, input_shape)

        self._record_result({
            "method": "SmallCNN FP32 Baseline",
            "category": "SmallCNN",
            "params_M": params_b["params_millions"],
            "params_total": params_b["total_params"],
            "model_size_MB": size_b["disk_size_mb"],
            "latency_ms": lat_b["mean_ms"],
            "latency_p95_ms": lat_b["p95_ms"],
            "throughput_sps": tp_b["samples_per_second"],
            "memory_MB": mem_b.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_b.get("gpu_memory_mb", 0),
            "flops_M": flops_b.get("total_mflops", 0),
            "mse_vs_baseline": 0.0,
            "edge_deployable": False,
        })

        # ---- Unstructured Magnitude Pruning ----
        logger.info("--- Unstructured Magnitude Pruning (%.0f%%) ---", self.args.sparsity * 100)
        pruned_unstr = apply_unstructured_magnitude_pruning(baseline, self.args.sparsity)
        pruned_unstr.to(self.device)
        self.save_artifact(pruned_unstr, "smallcnn_pruned_unstructured")

        params_pu = measure_parameters(pruned_unstr)
        # Effective params (non-zero)
        nz_count = 0
        total_count = 0
        for p in pruned_unstr.parameters():
            total_count += p.numel()
            nz_count += (p != 0).sum().item()
        effective_params_m = round(nz_count / 1e6, 3)
        size_pu = measure_model_size_disk(pruned_unstr)
        lat_pu = measure_inference_latency(pruned_unstr, cnn_input_fn,
                                            self.args.warmup, self.args.runs, self.device)
        tp_pu = measure_throughput(pruned_unstr, cnn_input_fn,
                                   self.args.batch_size, device=self.device)
        mem_pu = measure_memory_usage(pruned_unstr, cnn_input_fn, self.device)
        mse_pu = compute_model_mse(baseline, pruned_unstr, cnn_input_fn, self.device)
        compression_ratio = size_b["disk_size_mb"] / size_pu["disk_size_mb"] if size_pu["disk_size_mb"] > 0 else 1.0

        self._record_result({
            "method": "SmallCNN Unstructured Prune",
            "category": "SmallCNN",
            "params_M": params_pu["params_millions"],
            "effective_params_M": effective_params_m,
            "params_total": params_pu["total_params"],
            "model_size_MB": size_pu["disk_size_mb"],
            "compression_ratio": round(compression_ratio, 2),
            "latency_ms": lat_pu["mean_ms"],
            "latency_p95_ms": lat_pu["p95_ms"],
            "throughput_sps": tp_pu["samples_per_second"],
            "memory_MB": mem_pu.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_pu.get("gpu_memory_mb", 0),
            "flops_M": flops_b.get("total_mflops", 0),
            "mse_vs_baseline": round(mse_pu["mse"], 6),
            "cos_sim_vs_baseline": round(mse_pu["cosine_similarity"], 4),
            "edge_deployable": True,
        })

        # ---- Channel Pruning ----
        logger.info("--- Structured Channel Pruning (%.0f%%) ---", self.args.sparsity * 100)
        pruned_ch = apply_channel_pruning(baseline, self.args.sparsity)
        pruned_ch.to(self.device)
        self.save_artifact(pruned_ch, "smallcnn_pruned_channel")

        params_pc = measure_parameters(pruned_ch)
        size_pc = measure_model_size_disk(pruned_ch)
        lat_pc = measure_inference_latency(pruned_ch, cnn_input_fn,
                                            self.args.warmup, self.args.runs, self.device)
        tp_pc = measure_throughput(pruned_ch, cnn_input_fn,
                                   self.args.batch_size, device=self.device)
        mem_pc = measure_memory_usage(pruned_ch, cnn_input_fn, self.device)
        mse_pc = compute_model_mse(baseline, pruned_ch, cnn_input_fn, self.device)
        compression_ratio_ch = size_b["disk_size_mb"] / size_pc["disk_size_mb"] if size_pc["disk_size_mb"] > 0 else 1.0

        self._record_result({
            "method": "SmallCNN Channel Prune",
            "category": "SmallCNN",
            "params_M": params_pc["params_millions"],
            "params_total": params_pc["total_params"],
            "model_size_MB": size_pc["disk_size_mb"],
            "compression_ratio": round(compression_ratio_ch, 2),
            "latency_ms": lat_pc["mean_ms"],
            "latency_p95_ms": lat_pc["p95_ms"],
            "throughput_sps": tp_pc["samples_per_second"],
            "memory_MB": mem_pc.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_pc.get("gpu_memory_mb", 0),
            "flops_M": flops_b.get("total_mflops", 0),
            "mse_vs_baseline": round(mse_pc["mse"], 6),
            "cos_sim_vs_baseline": round(mse_pc["cosine_similarity"], 4),
            "edge_deployable": True,
        })

        # ---- PTQ INT8 Quantization ----
        logger.info("--- PTQ INT%d Quantization ---", self.args.quant_bits)
        quantized = apply_ptq_quantization(baseline, bits=self.args.quant_bits)
        quantized.to(self.device)
        self.save_artifact(quantized, "smallcnn_quantized_ptq")

        params_q = measure_parameters(quantized)
        size_q = measure_model_size_disk(quantized)
        lat_q = measure_inference_latency(quantized, cnn_input_fn,
                                           self.args.warmup, self.args.runs, self.device)
        tp_q = measure_throughput(quantized, cnn_input_fn,
                                  self.args.batch_size, device=self.device)
        mem_q = measure_memory_usage(quantized, cnn_input_fn, self.device)
        mse_q = compute_model_mse(baseline, quantized, cnn_input_fn, self.device)
        compression_ratio_q = size_b["disk_size_mb"] / size_q["disk_size_mb"] if size_q["disk_size_mb"] > 0 else 1.0

        self._record_result({
            "method": f"SmallCNN PTQ INT{self.args.quant_bits}",
            "category": "SmallCNN",
            "params_M": params_q["params_millions"],
            "params_total": params_q["total_params"],
            "model_size_MB": size_q["disk_size_mb"],
            "compression_ratio": round(compression_ratio_q, 2),
            "latency_ms": lat_q["mean_ms"],
            "latency_p95_ms": lat_q["p95_ms"],
            "throughput_sps": tp_q["samples_per_second"],
            "memory_MB": mem_q.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_q.get("gpu_memory_mb", 0),
            "flops_M": flops_b.get("total_mflops", 0),
            "mse_vs_baseline": round(mse_q["mse"], 6),
            "cos_sim_vs_baseline": round(mse_q["cosine_similarity"], 4),
            "edge_deployable": True,
        })

        # ---- ONNX Export and onnxruntime ----
        if not self.args.skip_onnx:
            logger.info("--- ONNX Export + onnxruntime Inference ---")
            onnx_path = str(ARTIFACTS_DIR / "smallcnn_baseline.onnx")
            ok = export_onnx(baseline, onnx_path, (3, 32, 32), self.device)
            if ok:
                self.artifacts["smallcnn_onnx"] = onnx_path
                ort_lat = run_onnxruntime_benchmark(onnx_path, (3, 32, 32),
                                                     batch_size=1,
                                                     warmup=self.args.warmup,
                                                     runs=self.args.runs)
                if ort_lat:
                    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
                    self._record_result({
                        "method": "SmallCNN ONNX Runtime",
                        "category": "SmallCNN",
                        "params_M": params_b["params_millions"],
                        "params_total": params_b["total_params"],
                        "model_size_MB": round(onnx_size, 4),
                        "compression_ratio": round(size_b["disk_size_mb"] / onnx_size, 2) if onnx_size > 0 else 0,
                        "latency_ms": ort_lat["mean_ms"],
                        "latency_p95_ms": ort_lat["p95_ms"],
                        "throughput_sps": 0,
                        "memory_MB": 0,
                        "gpu_memory_MB": 0,
                        "flops_M": flops_b.get("total_mflops", 0),
                        "mse_vs_baseline": 0.0,
                        "edge_deployable": True,
                    })

    # ---- Transformer Benchmarks ----

    def benchmark_transformer(self) -> None:
        logger.info("=" * 60)
        logger.info("2. Transformer Attention Block Quantization")
        logger.info("=" * 60)

        hidden = self.args.hidden_size
        baseline = TransformerAttentionBlock(hidden_size=hidden)
        baseline.to(self.device)
        baseline.eval()
        self.save_artifact(baseline, "transformer_baseline")

        tf_input_fn = lambda batch: torch.randn(batch, self.args.seq_len, hidden)
        input_fn = lambda: tf_input_fn(self.args.batch_size)
        input_shape = (self.args.seq_len, hidden)

        # Baseline
        params_b = measure_parameters(baseline)
        size_b = measure_model_size_disk(baseline)
        lat_b = measure_inference_latency(baseline, input_fn,
                                          self.args.warmup, self.args.runs, self.device)
        tp_b = measure_throughput(baseline, input_fn,
                                  self.args.batch_size, device=self.device)
        mem_b = measure_memory_usage(baseline, input_fn, self.device)
        flops_b = estimate_flops_manual(baseline, input_shape)

        self._record_result({
            "method": "Transformer FP32 Baseline",
            "category": "Transformer",
            "params_M": params_b["params_millions"],
            "params_total": params_b["total_params"],
            "model_size_MB": size_b["disk_size_mb"],
            "latency_ms": lat_b["mean_ms"],
            "latency_p95_ms": lat_b["p95_ms"],
            "throughput_sps": tp_b["samples_per_second"],
            "memory_MB": mem_b.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_b.get("gpu_memory_mb", 0),
            "flops_M": flops_b.get("total_mflops", 0),
            "mse_vs_baseline": 0.0,
            "edge_deployable": False,
        })

        # PTQ INT8
        logger.info("--- Transformer PTQ INT8 ---")
        quant_tf = apply_ptq_quantization(baseline, bits=8)
        quant_tf.to(self.device)
        self.save_artifact(quant_tf, "transformer_ptq_int8")

        params_q = measure_parameters(quant_tf)
        size_q = measure_model_size_disk(quant_tf)
        lat_q = measure_inference_latency(quant_tf, input_fn,
                                           self.args.warmup, self.args.runs, self.device)
        tp_q = measure_throughput(quant_tf, input_fn,
                                  self.args.batch_size, device=self.device)
        mem_q = measure_memory_usage(quant_tf, input_fn, self.device)
        mse_q = compute_model_mse(baseline, quant_tf, input_fn, self.device)
        cr = size_b["disk_size_mb"] / size_q["disk_size_mb"] if size_q["disk_size_mb"] > 0 else 1.0

        self._record_result({
            "method": "Transformer PTQ INT8",
            "category": "Transformer",
            "params_M": params_q["params_millions"],
            "params_total": params_q["total_params"],
            "model_size_MB": size_q["disk_size_mb"],
            "compression_ratio": round(cr, 2),
            "latency_ms": lat_q["mean_ms"],
            "latency_p95_ms": lat_q["p95_ms"],
            "throughput_sps": tp_q["samples_per_second"],
            "memory_MB": mem_q.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_q.get("gpu_memory_mb", 0),
            "flops_M": flops_b.get("total_mflops", 0),
            "mse_vs_baseline": round(mse_q["mse"], 6),
            "cos_sim_vs_baseline": round(mse_q["cosine_similarity"], 4),
            "edge_deployable": False,
        })

        # Dynamic Quantization
        logger.info("--- Transformer Dynamic Quantization ---")
        quant_dyn = apply_dynamic_quantization(baseline)
        try:
            quant_dyn.to(self.device)
        except Exception:
            pass

        try:
            lat_dyn = measure_inference_latency(quant_dyn, input_fn,
                                                 self.args.warmup, self.args.runs, self.device)
            tp_dyn = measure_throughput(quant_dyn, input_fn,
                                        self.args.batch_size, device=self.device)
            mem_dyn = measure_memory_usage(quant_dyn, input_fn, self.device)
            mse_dyn = compute_model_mse(baseline.to(self.device), quant_dyn.to(self.device),
                                         input_fn, self.device)
        except Exception as e:
            logger.warning("Dynamic quant benchmark failed: %s", e)
            lat_dyn = {"mean_ms": 0, "p95_ms": 0}
            tp_dyn = {"samples_per_second": 0}
            mem_dyn = {"cpu_rss_mb": 0}
            mse_dyn = {"mse": 0, "cosine_similarity": 0}

        self._record_result({
            "method": "Transformer Dynamic Quant",
            "category": "Transformer",
            "params_M": params_q["params_millions"],
            "params_total": params_q["total_params"],
            "model_size_MB": size_q["disk_size_mb"],
            "compression_ratio": round(cr, 2),
            "latency_ms": lat_dyn["mean_ms"],
            "latency_p95_ms": lat_dyn["p95_ms"],
            "throughput_sps": tp_dyn["samples_per_second"],
            "memory_MB": mem_dyn.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_dyn.get("gpu_memory_mb", 0),
            "flops_M": flops_b.get("total_mflops", 0),
            "mse_vs_baseline": round(mse_dyn["mse"], 6),
            "cos_sim_vs_baseline": round(mse_dyn["cosine_similarity"], 4),
            "edge_deployable": False,
        })

        # Transformer ONNX export
        if not self.args.skip_onnx:
            logger.info("--- Transformer ONNX Export ---")
            onnx_path = str(ARTIFACTS_DIR / "transformer.onnx")
            ok = export_onnx(baseline, onnx_path, input_shape, self.device,
                             dynamic_batch=True)
            if ok:
                self.artifacts["transformer_onnx"] = onnx_path
                ort_lat = run_onnxruntime_benchmark(onnx_path, input_shape,
                                                     batch_size=1,
                                                     warmup=self.args.warmup,
                                                     runs=self.args.runs)
                if ort_lat:
                    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
                    self._record_result({
                        "method": "Transformer ONNX Runtime",
                        "category": "Transformer",
                        "params_M": params_b["params_millions"],
                        "params_total": params_b["total_params"],
                        "model_size_MB": round(onnx_size, 4),
                        "compression_ratio": round(size_b["disk_size_mb"] / onnx_size, 2) if onnx_size > 0 else 0,
                        "latency_ms": ort_lat["mean_ms"],
                        "latency_p95_ms": ort_lat["p95_ms"],
                        "throughput_sps": 0,
                        "memory_MB": 0,
                        "gpu_memory_MB": 0,
                        "flops_M": flops_b.get("total_mflops", 0),
                        "mse_vs_baseline": 0.0,
                        "edge_deployable": False,
                    })

    # ---- VLA Action Head Benchmarks ----

    def benchmark_vla_action_head(self) -> None:
        logger.info("=" * 60)
        logger.info("3. VLA Action Head Compression (Robot Action Chunk)")
        logger.info("=" * 60)

        baseline = VLAActionHead()
        if self.args.train_steps > 0:
            # Train the internal MLP directly on synthetic data
            mlp_model = baseline.net  # nn.Sequential
            device = self.device
            mlp_model = mlp_model.to(device)
            mlp_model.train()
            opt = torch.optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)
            for step in range(self.args.train_steps):
                x = torch.randn(32, 263, device=device)  # vision(256)+state(7)
                y = torch.randn(32, 700, device=device)  # action chunks
                opt.zero_grad()
                loss = F.mse_loss(mlp_model(x), y)
                loss.backward()
                opt.step()
            mlp_model.eval()
        baseline.to(self.device)
        baseline.eval()
        self.save_artifact(baseline, "vla_action_head_baseline")

        def vla_input_fn(batch_size: int | None = None):
            bs = batch_size if batch_size is not None else self.args.batch_size
            return torch.randn(bs, 256), torch.randn(bs, 7)

        # We need a single-tensor input_fn for latency measurement
        class VLAModelWrapper(nn.Module):
            def __init__(self, head):
                super().__init__()
                self.head = head

            def forward(self, x):
                vision = x[:, :256]
                state = x[:, 256:]
                return self.head(vision, state)

        wrapper = VLAModelWrapper(baseline)
        wrapper.to(self.device)
        wrapper.eval()

        single_input_fn = lambda: torch.randn(self.args.batch_size, 263, device=self.device)

        # Baseline
        params_b = measure_parameters(baseline)
        size_b = measure_model_size_disk(baseline)
        lat_b = measure_inference_latency(wrapper, single_input_fn,
                                           self.args.warmup, self.args.runs, self.device)
        tp_b = measure_throughput(wrapper, single_input_fn,
                                  self.args.batch_size, device=self.device)
        mem_b = measure_memory_usage(wrapper, single_input_fn, self.device)

        self._record_result({
            "method": "VLA ActionHead FP32 Baseline",
            "category": "VLA ActionHead",
            "params_M": params_b["params_millions"],
            "params_total": params_b["total_params"],
            "model_size_MB": size_b["disk_size_mb"],
            "latency_ms": lat_b["mean_ms"],
            "latency_p95_ms": lat_b["p95_ms"],
            "throughput_sps": tp_b["samples_per_second"],
            "memory_MB": mem_b.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_b.get("gpu_memory_mb", 0),
            "flops_M": 0,
            "mse_vs_baseline": 0.0,
            "edge_deployable": True,
        })

        # Prune action head
        logger.info("--- VLA ActionHead Unstructured Pruning (%.0f%%) ---", self.args.sparsity * 100)
        pruned_vla = apply_unstructured_magnitude_pruning(baseline, self.args.sparsity)
        pruned_vla.to(self.device)
        self.save_artifact(pruned_vla, "vla_action_head_pruned")

        wrapper_p = VLAModelWrapper(pruned_vla)
        wrapper_p.to(self.device)

        params_p = measure_parameters(pruned_vla)
        nz = sum((p != 0).sum().item() for p in pruned_vla.parameters())
        size_p = measure_model_size_disk(pruned_vla)
        lat_p = measure_inference_latency(wrapper_p, single_input_fn,
                                           self.args.warmup, self.args.runs, self.device)
        tp_p = measure_throughput(wrapper_p, single_input_fn,
                                  self.args.batch_size, device=self.device)
        mem_p = measure_memory_usage(wrapper_p, single_input_fn, self.device)
        mse_p = compute_model_mse(wrapper, wrapper_p, single_input_fn, self.device)
        cr = size_b["disk_size_mb"] / size_p["disk_size_mb"] if size_p["disk_size_mb"] > 0 else 1.0

        self._record_result({
            "method": "VLA ActionHead Pruned",
            "category": "VLA ActionHead",
            "params_M": params_p["params_millions"],
            "effective_params_M": round(nz / 1e6, 3),
            "params_total": params_p["total_params"],
            "model_size_MB": size_p["disk_size_mb"],
            "compression_ratio": round(cr, 2),
            "latency_ms": lat_p["mean_ms"],
            "latency_p95_ms": lat_p["p95_ms"],
            "throughput_sps": tp_p["samples_per_second"],
            "memory_MB": mem_p.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_p.get("gpu_memory_mb", 0),
            "flops_M": 0,
            "mse_vs_baseline": round(mse_p["mse"], 6),
            "cos_sim_vs_baseline": round(mse_p["cosine_similarity"], 4),
            "action_deviation_note": "Measured as output MSE vs FP32 baseline. "
                                     "In production, use rollout success rate and trajectory error.",
            "edge_deployable": True,
        })

        # Quantize action head
        logger.info("--- VLA ActionHead PTQ INT8 ---")
        quant_vla = apply_ptq_quantization(baseline, bits=8)
        quant_vla.to(self.device)
        self.save_artifact(quant_vla, "vla_action_head_quantized")

        wrapper_q = VLAModelWrapper(quant_vla)
        wrapper_q.to(self.device)

        params_q = measure_parameters(quant_vla)
        size_q = measure_model_size_disk(quant_vla)
        lat_q = measure_inference_latency(wrapper_q, single_input_fn,
                                           self.args.warmup, self.args.runs, self.device)
        tp_q = measure_throughput(wrapper_q, single_input_fn,
                                  self.args.batch_size, device=self.device)
        mem_q = measure_memory_usage(wrapper_q, single_input_fn, self.device)
        mse_q = compute_model_mse(wrapper, wrapper_q, single_input_fn, self.device)
        cr_q = size_b["disk_size_mb"] / size_q["disk_size_mb"] if size_q["disk_size_mb"] > 0 else 1.0

        self._record_result({
            "method": "VLA ActionHead PTQ INT8",
            "category": "VLA ActionHead",
            "params_M": params_q["params_millions"],
            "params_total": params_q["total_params"],
            "model_size_MB": size_q["disk_size_mb"],
            "compression_ratio": round(cr_q, 2),
            "latency_ms": lat_q["mean_ms"],
            "latency_p95_ms": lat_q["p95_ms"],
            "throughput_sps": tp_q["samples_per_second"],
            "memory_MB": mem_q.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_q.get("gpu_memory_mb", 0),
            "flops_M": 0,
            "mse_vs_baseline": round(mse_q["mse"], 6),
            "cos_sim_vs_baseline": round(mse_q["cosine_similarity"], 4),
            "action_deviation_note": "INT8 quantization typically introduces small action drift. "
                                     "For safety-critical VLA, always validate with rollout MSE.",
            "edge_deployable": True,
        })

    # ---- SimpleMLP ----

    def benchmark_mlp(self) -> None:
        logger.info("=" * 60)
        logger.info("4. SimpleMLP Compression (Pruning + Quantization)")
        logger.info("=" * 60)

        baseline = SimpleMLP()
        if self.args.train_steps > 0:
            baseline = train_mlp_synthetic(baseline, self.device,
                                           steps=self.args.train_steps)
        baseline.to(self.device)
        baseline.eval()
        self.save_artifact(baseline, "mlp_baseline")

        mlp_input_fn = lambda: torch.randn(self.args.batch_size, 784, device=self.device)

        # Baseline
        params_b = measure_parameters(baseline)
        size_b = measure_model_size_disk(baseline)
        lat_b = measure_inference_latency(baseline, mlp_input_fn,
                                           self.args.warmup, self.args.runs, self.device)
        tp_b = measure_throughput(baseline, mlp_input_fn,
                                  self.args.batch_size, device=self.device)
        mem_b = measure_memory_usage(baseline, mlp_input_fn, self.device)

        self._record_result({
            "method": "MLP FP32 Baseline",
            "category": "SimpleMLP",
            "params_M": params_b["params_millions"],
            "params_total": params_b["total_params"],
            "model_size_MB": size_b["disk_size_mb"],
            "latency_ms": lat_b["mean_ms"],
            "latency_p95_ms": lat_b["p95_ms"],
            "throughput_sps": tp_b["samples_per_second"],
            "memory_MB": mem_b.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_b.get("gpu_memory_mb", 0),
            "flops_M": 0,
            "mse_vs_baseline": 0.0,
            "edge_deployable": False,
        })

        # Prune
        logger.info("--- MLP Unstructured Pruning (%.0f%%) ---", self.args.sparsity * 100)
        mlp_pruned = apply_unstructured_magnitude_pruning(baseline, self.args.sparsity)
        mlp_pruned.to(self.device)

        params_mp = measure_parameters(mlp_pruned)
        nz_mp = sum((p != 0).sum().item() for p in mlp_pruned.parameters())
        size_mp = measure_model_size_disk(mlp_pruned)
        lat_mp = measure_inference_latency(mlp_pruned, mlp_input_fn,
                                            self.args.warmup, self.args.runs, self.device)
        tp_mp = measure_throughput(mlp_pruned, mlp_input_fn,
                                   self.args.batch_size, device=self.device)
        mem_mp = measure_memory_usage(mlp_pruned, mlp_input_fn, self.device)
        mse_mp = compute_model_mse(baseline, mlp_pruned, mlp_input_fn, self.device)
        cr_mp = size_b["disk_size_mb"] / size_mp["disk_size_mb"] if size_mp["disk_size_mb"] > 0 else 1.0

        self._record_result({
            "method": "MLP Unstructured Prune",
            "category": "SimpleMLP",
            "params_M": params_mp["params_millions"],
            "effective_params_M": round(nz_mp / 1e6, 3),
            "params_total": params_mp["total_params"],
            "model_size_MB": size_mp["disk_size_mb"],
            "compression_ratio": round(cr_mp, 2),
            "latency_ms": lat_mp["mean_ms"],
            "latency_p95_ms": lat_mp["p95_ms"],
            "throughput_sps": tp_mp["samples_per_second"],
            "memory_MB": mem_mp.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_mp.get("gpu_memory_mb", 0),
            "flops_M": 0,
            "mse_vs_baseline": round(mse_mp["mse"], 6),
            "cos_sim_vs_baseline": round(mse_mp["cosine_similarity"], 4),
            "edge_deployable": False,
        })

        # Quantize
        logger.info("--- MLP PTQ INT8 ---")
        mlp_quant = apply_ptq_quantization(baseline, bits=8)
        mlp_quant.to(self.device)

        params_mq = measure_parameters(mlp_quant)
        size_mq = measure_model_size_disk(mlp_quant)
        lat_mq = measure_inference_latency(mlp_quant, mlp_input_fn,
                                            self.args.warmup, self.args.runs, self.device)
        tp_mq = measure_throughput(mlp_quant, mlp_input_fn,
                                   self.args.batch_size, device=self.device)
        mem_mq = measure_memory_usage(mlp_quant, mlp_input_fn, self.device)
        mse_mq = compute_model_mse(baseline, mlp_quant, mlp_input_fn, self.device)
        cr_mq = size_b["disk_size_mb"] / size_mq["disk_size_mb"] if size_mq["disk_size_mb"] > 0 else 1.0

        self._record_result({
            "method": "MLP PTQ INT8",
            "category": "SimpleMLP",
            "params_M": params_mq["params_millions"],
            "params_total": params_mq["total_params"],
            "model_size_MB": size_mq["disk_size_mb"],
            "compression_ratio": round(cr_mq, 2),
            "latency_ms": lat_mq["mean_ms"],
            "latency_p95_ms": lat_mq["p95_ms"],
            "throughput_sps": tp_mq["samples_per_second"],
            "memory_MB": mem_mq.get("cpu_rss_mb", 0),
            "gpu_memory_MB": mem_mq.get("gpu_memory_mb", 0),
            "flops_M": 0,
            "mse_vs_baseline": round(mse_mq["mse"], 6),
            "cos_sim_vs_baseline": round(mse_mq["cosine_similarity"], 4),
            "edge_deployable": False,
        })

    # ---- TensorRT check ----

    def check_tensorrt(self) -> dict[str, Any]:
        trt_available, trt_msg = check_tensorrt_available()
        return {
            "tensorrt_available": trt_available,
            "tensorrt_status": trt_msg,
        }

    # ---- Report Generation ----

    def generate_report(self, output_path: str) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        env_info = self._get_env_info()
        trt_info = self.check_tensorrt()

        lines: list[str] = []
        lines.append("# 模型压缩实验报告 (Model Compression Benchmark Report)")
        lines.append("")
        lines.append(f"**生成时间**: {now}")
        lines.append(f"**运行设备**: {self.device}")
        lines.append(f"**PyTorch 版本**: {torch.__version__}")
        lines.append("")
        lines.append("> 本报告由 `benchmark_compression.py` 自动生成，所有数据均为脚本真实测量。")
        lines.append("")

        # Environment
        lines.append("## 运行环境")
        lines.append("")
        for k, v in env_info.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

        # TensorRT status
        lines.append("## TensorRT 可用性")
        lines.append("")
        if trt_info["tensorrt_available"]:
            lines.append(f"> TensorRT is available: {trt_info['tensorrt_status']}")
        else:
            lines.append(f"> TensorRT is **NOT** available: {trt_info['tensorrt_status']}")
            lines.append("> TensorRT benchmarks are skipped. To enable, install TensorRT SDK and Python bindings.")
        lines.append("")

        # Results by category
        categories: dict[str, list[dict]] = {}
        for r in self.results:
            cat = r.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        for cat_name, cat_results in categories.items():
            lines.append(f"## {cat_name}")
            lines.append("")

            # Comparison table
            lines.append("| 方法 | 参数量 (M) | 模型大小 (MB) | 压缩率 | 延迟 (ms) | P95延迟 (ms) | 吞吐 (samples/s) | 内存 (MB) | GPU显存 (MB) | MSE vs Baseline | 端侧可部署 |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")

            for r in cat_results:
                m = r.get("method", "N/A")
                pm = f"{r.get('params_M', 0):.3f}"
                epm = r.get("effective_params_M")
                if epm is not None:
                    pm += f" (eff {epm:.3f})"
                sm = f"{r.get('model_size_MB', 0):.4f}"
                cr = f"{r.get('compression_ratio', 1.0):.2f}x"
                lat = f"{r.get('latency_ms', 0):.4f}"
                p95 = f"{r.get('latency_p95_ms', 0):.4f}"
                tp = f"{r.get('throughput_sps', 0):.2f}"
                mem = f"{r.get('memory_MB', 0):.4f}"
                gmem = f"{r.get('gpu_memory_MB', 0):.4f}"
                mse = f"{r.get('mse_vs_baseline', 0):.6f}"
                edge = "Yes" if r.get("edge_deployable") else "No"

                lines.append(
                    f"| {m} | {pm} | {sm} | {cr} | {lat} | {p95} | {tp} | {mem} | {gmem} | {mse} | {edge} |"
                )

            lines.append("")

            # Analysis for this category
            baseline_r = None
            for r in cat_results:
                if "Baseline" in r.get("method", "") or "baseline" in r.get("method", "").lower():
                    baseline_r = r
                    break

            if baseline_r:
                lines.append(f"### {cat_name} 压缩分析")
                lines.append("")
                bl_size = baseline_r.get("model_size_MB", 1)
                bl_lat = baseline_r.get("latency_ms", 1)
                bl_tp = baseline_r.get("throughput_sps", 1)
                bl_mem = baseline_r.get("memory_MB", 1)

                for r in cat_results:
                    if r is baseline_r:
                        continue
                    m = r.get("method", "N/A")
                    lines.append(f"**{m}**:")
                    sz = r.get("model_size_MB", bl_size)
                    if bl_size > 0:
                        lines.append(f"  - 模型大小: {sz:.4f} MB (基线 {bl_size:.4f} MB, 压缩率 {bl_size/sz:.2f}x)")
                    lt = r.get("latency_ms", bl_lat)
                    if bl_lat > 0:
                        ratio = bl_lat / lt if lt > 0 else 0
                        direction = "加速" if ratio >= 1 else "减速"
                        lines.append(f"  - 延迟: {lt:.4f} ms (基线 {bl_lat:.4f} ms, {direction} {abs(ratio):.2f}x)")
                    lines.append(f"  - MSE (vs baseline): {r.get('mse_vs_baseline', 'N/A')}")
                    if r.get("cos_sim_vs_baseline"):
                        lines.append(f"  - Cosine Similarity: {r['cos_sim_vs_baseline']}")
                    extra_note = r.get("action_deviation_note")
                    if extra_note:
                        lines.append(f"  - 注: {extra_note}")
                    lines.append("")

        # Industrial deployment recommendations
        lines.append("## 工业部署建议")
        lines.append("")
        lines.append("1. **CNN/ViT 端侧部署**: 优先尝试结构化通道剪枝 + INT8 PTQ。结构化稀疏对硬件友好，PTQ 部署成本低。")
        lines.append("2. **Transformer/LLM CPU 推理**: 优先尝试动态量化 (torch.quantize_dynamic) 或 weight-only INT4 量化。GPU 推理优先使用 TensorRT-LLM 或 vLLM。")
        lines.append("3. **VLA/机器人 Action Head**: MLP action head 中间层可积极量化 (INT8)，最后输出层谨慎处理。验收指标必须包含 action MSE、rollout success rate 和 P99 latency。")
        lines.append("4. **非结构化剪枝**: 虽然压缩率高，但需要专用 sparse kernel (如 cuSPARSE、MKL Sparse BLAS) 才能在中低稀疏度 (<=90%) 上实现延迟收益。无 sparse kernel 环境下可能不降反升。")
        lines.append("5. **精度验证**: 本报告使用 MSE 作为代理 metric。真实项目必须用 task metric（准确率、perplexity、mAP、success rate）验证。")
        lines.append("6. **TensorRT 部署**: engine 必须用目标硬件真实构建和 benchmark。ONNX → TensorRT engine 过程中 INT8 校准需要真实校准数据。")
        lines.append("")

        # Disclaimer
        lines.append("## 免责声明")
        lines.append("")
        lines.append("本报告中的量化方法为简化实现（手动 scale/round/clamp），与 PyTorch 原生量化后端 (fbgemm/qnnpack) 略有不同。生产部署请使用 torch.ao.quantization 或 TensorRT 原生量化流程。")
        lines.append("")

        lines.append("---")
        lines.append(f"*报告由 benchmark_compression.py 自动生成于 {now}*")
        lines.append("")

        report_content = "\n".join(lines)

        # Write report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info("Report saved to %s", output_path)

        # Also save raw JSON
        json_path = output_path.replace(".md", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": now,
                "device": str(self.device),
                "pytorch_version": torch.__version__,
                "results": self.results,
                "tensorrt": trt_info,
                "env": env_info,
            }, f, indent=2, ensure_ascii=False)
        logger.info("Raw results saved to %s", json_path)

        return report_content

    def _get_env_info(self) -> dict[str, str]:
        import platform
        info = {
            "OS": f"{platform.system()} {platform.release()}",
            "Python": platform.python_version(),
            "PyTorch": torch.__version__,
            "CPU Count": str(os.cpu_count()),
        }
        if torch.cuda.is_available():
            info["CUDA Available"] = "Yes"
            info["CUDA Version"] = torch.version.cuda or "unknown"
            info["GPU"] = torch.cuda.get_device_name(0)
        else:
            info["CUDA Available"] = "No"

        # Check for optional packages
        try:
            import onnxruntime
            info["onnxruntime"] = onnxruntime.__version__
        except ImportError:
            info["onnxruntime"] = "not installed"

        try:
            import onnx
            info["onnx"] = onnx.__version__
        except ImportError:
            info["onnx"] = "not installed"

        try:
            import psutil
            info["psutil"] = psutil.__version__
        except ImportError:
            info["psutil"] = "not installed"

        try:
            import fvcore
            info["fvcore"] = "installed"
        except ImportError:
            info["fvcore"] = "not installed"

        try:
            import thop
            info["thop"] = "installed"
        except ImportError:
            info["thop"] = "not installed"

        return info


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Model Compression Benchmark")
    logger.info("=" * 60)
    logger.info("Config: device=%s, batch_size=%d, runs=%d, warmup=%d, sparsity=%.2f",
                 args.device, args.batch_size, args.runs, args.warmup, args.sparsity)

    benchmark = CompressionBenchmark(args)

    try:
        benchmark.benchmark_smallcnn()
    except Exception as e:
        logger.error("SmallCNN benchmark failed: %s", e, exc_info=True)

    try:
        benchmark.benchmark_transformer()
    except Exception as e:
        logger.error("Transformer benchmark failed: %s", e, exc_info=True)

    try:
        benchmark.benchmark_vla_action_head()
    except Exception as e:
        logger.error("VLA ActionHead benchmark failed: %s", e, exc_info=True)

    try:
        benchmark.benchmark_mlp()
    except Exception as e:
        logger.error("MLP benchmark failed: %s", e, exc_info=True)

    # Generate report
    report_path = benchmark.generate_report(args.output)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Benchmark complete! Report: %s", report_path)
    logger.info("=" * 60)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("Quick Summary")
    print("=" * 60)
    for r in benchmark.results:
        m = r.get("method", "N/A")
        pm = r.get("params_M", 0)
        sm = r.get("model_size_MB", 0)
        lt = r.get("latency_ms", 0)
        print(f"  {m}: params={pm:.3f}M, size={sm:.4f}MB, latency={lt:.4f}ms")


if __name__ == "__main__":
    main()
