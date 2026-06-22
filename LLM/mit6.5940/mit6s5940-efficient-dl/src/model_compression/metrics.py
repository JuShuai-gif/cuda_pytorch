"""
Measurement utilities for model compression benchmarks.

Provides:
- Parameter counting (native + torchinfo)
- FLOPs/MACs estimation (manual, fvcore, thop, torchprofile, calflops)
- Inference latency (time.perf_counter + torch.utils.benchmark)
- Memory usage (psutil, torch.cuda, pynvml)
- GPU monitoring (pynvml: utilization, temperature, power)
- Model file size on disk
- Throughput measurement
- Model comparison (MSE, cosine similarity, KL divergence, SNR)
- TensorBoard / W&B logging helpers
- PyTorch profiler integration
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Library availability flags (lazy detection)
# ============================================================

_LIB_AVAIL: dict[str, bool] = {}


def _check_lib(name: str, import_path: str) -> bool:
    """Lazily check and cache library availability."""
    if name not in _LIB_AVAIL:
        try:
            __import__(import_path)
            _LIB_AVAIL[name] = True
        except ImportError:
            _LIB_AVAIL[name] = False
    return _LIB_AVAIL[name]


def _has_psutil() -> bool: return _check_lib("psutil", "psutil")
def _has_pynvml() -> bool: return _check_lib("pynvml", "pynvml")
def _has_torchprofile() -> bool: return _check_lib("torchprofile", "torchprofile")
def _has_fvcore() -> bool: return _check_lib("fvcore", "fvcore.nn")
def _has_thop() -> bool: return _check_lib("thop", "thop")
def _has_calflops() -> bool: return _check_lib("calflops", "calflops")
def _has_torchinfo() -> bool: return _check_lib("torchinfo", "torchinfo")
def _has_wandb() -> bool: return _check_lib("wandb", "wandb")

# ============================================================
# Parameter counting
# ============================================================


def measure_parameters(model: nn.Module) -> dict[str, Any]:
    """Count model parameters with breakdown.

    Returns:
        total_params, trainable_params, params_millions,
        fp32_size_mb, fp16_size_mb, bf16_size_mb,
        int8_size_mb, int4_size_mb
    """
    total = 0
    trainable = 0
    per_type: dict[str, int] = {}

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        tname = type(p).__name__
        per_type[tname] = per_type.get(tname, 0) + n

    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
        "params_millions": round(total / 1e6, 3),
        "fp32_size_mb": round(total * 4 / (1024 * 1024), 3),
        "fp16_size_mb": round(total * 2 / (1024 * 1024), 3),
        "bf16_size_mb": round(total * 2 / (1024 * 1024), 3),
        "int8_size_mb": round(total * 1 / (1024 * 1024), 3),
        "int4_size_mb": round(total * 0.5 / (1024 * 1024), 3),
        "param_types": per_type,
    }


def torchinfo_summary(
    model: nn.Module,
    input_shape: tuple[int, ...],
    batch_dim: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any] | None:
    """Generate detailed model summary using torchinfo (if available).

    Returns None if torchinfo is not installed.

    Args:
        model: PyTorch model
        input_shape: input tensor shape WITHOUT batch dim, e.g. (3, 32, 32)
        batch_dim: where to insert batch dimension (0 for NCHW, 1 for transformer)
        device: device for dummy input
    """
    if not _has_torchinfo():
        return None

    from torchinfo import summary

    if device is None:
        device = next(model.parameters()).device

    # Build input_size tuple: (batch, *shape) or (*shape[:1], batch, *shape[1:])
    if batch_dim == 0:
        input_size = (1,) + input_shape
    else:
        input_size = input_shape[:batch_dim] + (1,) + input_shape[batch_dim:]

    try:
        s = summary(model, input_size=input_size, device=device, verbose=0)
        return {
            "total_params": s.total_params,
            "trainable_params": s.trainable_params,
            "total_mult_adds": s.total_mult_adds,
            "total_flops_est": s.total_mult_adds * 2.0 if s.total_mult_adds else None,
            "input_size_mb": round(s.total_input / (1024 * 1024), 3),
            "fwd_bwd_size_mb": round(s.total_output_bytes / (1024 * 1024), 3),
            "estimated_total_size_mb": round(
                (s.total_input + s.total_output_bytes + s.total_param_bytes)
                / (1024 * 1024), 3
            ),
            "library": "torchinfo",
        }
    except Exception:
        return None


# ============================================================
# FLOPs / MACs estimation (multi-library)
# ============================================================


def estimate_flops_manual(
    model: nn.Module, input_shape: tuple[int, ...]
) -> dict[str, Any]:
    """Manually estimate FLOPs for Conv2d/Linear/BatchNorm layers.

    Uses forward hooks to capture output shapes, then applies standard
    FLOPs formulas. This is library-independent and always works.
    """
    feature_maps: dict[str, tuple[int, ...]] = {}
    hooks: list[Any] = []

    def _hook(name: str):
        def _fn(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                feature_maps[name] = tuple(out.shape)

        return _fn

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(_hook(name)))

    device = next(model.parameters()).device
    with torch.no_grad():
        dummy = torch.randn(1, *input_shape, device=device)
        model.eval()
        _ = model(dummy)

    for h in hooks:
        h.remove()

    total_flops = 0
    per_layer: dict[str, int] = {}

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            out_shape = feature_maps.get(name)
            if out_shape is None:
                continue
            k_h, k_w = m.kernel_size
            flops = (
                2 * k_h * k_w * (m.in_channels // m.groups)
                * m.out_channels * out_shape[2] * out_shape[3]
            )
            if m.bias is not None:
                flops += m.out_channels * out_shape[2] * out_shape[3]
            total_flops += flops
            per_layer[name] = int(flops)

        elif isinstance(m, nn.ConvTranspose2d):
            out_shape = feature_maps.get(name)
            if out_shape is None:
                continue
            k_h, k_w = m.kernel_size
            flops = (
                2 * k_h * k_w * (m.in_channels // m.groups)
                * m.out_channels * out_shape[2] * out_shape[3]
            )
            if m.bias is not None:
                flops += m.out_channels * out_shape[2] * out_shape[3]
            total_flops += flops
            per_layer[name] = int(flops)

        elif isinstance(m, nn.Linear):
            flops = 2 * m.in_features * m.out_features
            if m.bias is not None:
                flops += m.out_features
            total_flops += flops
            per_layer[name] = int(flops)

    return {
        "total_flops": int(total_flops),
        "total_mflops": round(total_flops / 1e6, 3),
        "total_gflops": round(total_flops / 1e9, 6),
        "per_layer": per_layer,
        "method": "manual_hooks",
    }


def estimate_flops_fvcore(
    model: nn.Module, input_shape: tuple[int, ...]
) -> dict[str, Any] | None:
    """Estimate FLOPs using fvcore (Meta's FlopCountAnalysis)."""
    if not _has_fvcore():
        return None

    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    device = next(model.parameters()).device
    model.eval()
    dummy = torch.randn(1, *input_shape, device=device)

    try:
        flops = FlopCountAnalysis(model, dummy)
        return {
            "total_flops": int(flops.total()),
            "total_mflops": round(flops.total() / 1e6, 3),
            "total_gflops": round(flops.total() / 1e9, 6),
            "by_operator": flops.by_operator(),
            "method": "fvcore",
        }
    except Exception:
        return None


def estimate_flops_thop(
    model: nn.Module, input_shape: tuple[int, ...]
) -> dict[str, Any] | None:
    """Estimate FLOPs using thop (pytorch-OpCounter)."""
    if not _has_thop():
        return None

    from thop import profile

    device = next(model.parameters()).device
    model.eval()
    dummy = torch.randn(1, *input_shape, device=device)

    try:
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return {
            "total_macs": int(flops),
            "total_flops": int(flops) * 2,
            "total_mflops": round(flops * 2 / 1e6, 3),
            "params_from_thop": int(params),
            "method": "thop",
        }
    except Exception:
        return None


def estimate_flops_torchprofile(
    model: nn.Module, input_shape: tuple[int, ...]
) -> dict[str, Any] | None:
    """Estimate MACs using torchprofile."""
    if not _has_torchprofile():
        return None

    from torchprofile import profile_macs

    device = next(model.parameters()).device
    model.eval()
    dummy = torch.randn(1, *input_shape, device=device)

    try:
        macs = profile_macs(model, dummy)
        return {
            "total_macs": int(macs),
            "total_flops": int(macs) * 2,
            "total_mflops": round(macs * 2 / 1e6, 3),
            "method": "torchprofile",
        }
    except Exception:
        return None


def estimate_flops_calflops(
    model: nn.Module, input_shape: tuple[int, ...]
) -> dict[str, Any] | None:
    """Estimate FLOPs using calflops (HuggingFace-aware)."""
    if not _has_calflops():
        return None

    from calflops import calculate_flops

    device = next(model.parameters()).device
    model.eval()

    try:
        flops, params, _ = calculate_flops(
            model,
            input_shape=(1,) + input_shape,
            output_as_string=False,
            print_detailed=False,
        )
        return {
            "total_flops": int(flops),
            "total_mflops": round(flops / 1e6, 3),
            "params_from_calflops": int(params),
            "method": "calflops",
        }
    except Exception:
        return None


def estimate_flops_all(
    model: nn.Module, input_shape: tuple[int, ...]
) -> dict[str, Any]:
    """Estimate FLOPs using ALL available libraries.

    Returns a dict with results from each successful method, plus
    a summary with the average and range.
    """
    results: dict[str, Any] = {}
    flops_values: list[float] = []

    # Always try manual (no dependencies)
    manual = estimate_flops_manual(model, input_shape)
    results["manual"] = manual
    flops_values.append(manual["total_flops"])

    # Try each library
    for name, fn in [
        ("fvcore", estimate_flops_fvcore),
        ("thop", estimate_flops_thop),
        ("torchprofile", estimate_flops_torchprofile),
        ("calflops", estimate_flops_calflops),
    ]:
        r = fn(model, input_shape)
        if r is not None:
            results[name] = r
            flops_values.append(r["total_flops"])

    if len(flops_values) >= 2:
        arr = np.array(flops_values)
        results["summary"] = {
            "num_methods": len(flops_values),
            "mean_mflops": round(float(arr.mean()) / 1e6, 3),
            "std_mflops": round(float(arr.std()) / 1e6, 3),
            "min_mflops": round(float(arr.min()) / 1e6, 3),
            "max_mflops": round(float(arr.max()) / 1e6, 3),
            "spread_pct": round(
                (arr.max() - arr.min()) / arr.mean() * 100, 2
            ) if arr.mean() > 0 else 0,
            "note": (
                "Different libraries count BatchNorm/ReLU/Pool differently. "
                "Spread < 10% is normal; cite your library in publications."
            ),
        }

    return results


# ============================================================
# Inference latency & throughput
# ============================================================


def measure_inference_latency(
    model: nn.Module,
    input_fn,
    warmup_runs: int = 10,
    measure_runs: int = 100,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Measure inference latency with warmup via time.perf_counter.

    Returns dict with mean_ms, median_ms, std_ms, min_ms, max_ms,
    p50_ms, p95_ms, p99_ms, plus per-run latencies list.

    For more rigorous benchmarking, use measure_latency_benchmark()
    which uses torch.utils.benchmark.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            inp = input_fn()
            if isinstance(inp, torch.Tensor):
                inp = inp.to(device)
            _ = model(inp)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    latencies: list[float] = []
    for _ in range(measure_runs):
        with torch.no_grad():
            inp = input_fn()
            if isinstance(inp, torch.Tensor):
                inp = inp.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "mean_ms": round(float(arr.mean()), 4),
        "median_ms": round(float(np.median(arr)), 4),
        "std_ms": round(float(arr.std()), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "method": "time.perf_counter",
        "raw_latencies_ms": latencies,
    }


def measure_latency_benchmark(
    model: nn.Module,
    input_fn,
    num_runs: int = 100,
    device: torch.device | None = None,
    label: str = "inference",
) -> dict[str, Any]:
    """Measure inference latency using torch.utils.benchmark (more accurate).

    Uses cudaEvents for GPU timing and provides sub-millisecond precision.
    Falls back to measure_inference_latency if benchmark is not suitable.
    """
    try:
        from torch.utils.benchmark import Timer

        if device is None:
            device = next(model.parameters()).device

        model.eval()

        # Define the workload
        def _work():
            with torch.no_grad():
                inp = input_fn()
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(device)
                return model(inp)

        timer = Timer(
            stmt="_work()",
            globals={"_work": _work},
            label=label,
            num_threads=torch.get_num_threads(),
        )

        # First warmup
        _work()

        measurement = timer.blocked_autorange(min_run_time=0.5)
        latencies_ms = [t * 1000 for t in measurement.raw_times]
        arr = np.array(latencies_ms)

        return {
            "mean_ms": round(measurement.mean * 1000, 4),
            "median_ms": round(measurement.median * 1000, 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
            "num_runs": measurement.number,
            "method": "torch.utils.benchmark",
        }
    except Exception:
        return measure_inference_latency(model, input_fn, measure_runs=num_runs)


def measure_throughput(
    model: nn.Module,
    input_fn,
    batch_size: int,
    num_batches: int = 50,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Measure inference throughput in samples/second."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Warmup
    with torch.no_grad():
        inp = input_fn()
        if isinstance(inp, torch.Tensor):
            inp = inp.to(device)
        _ = model(inp)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_samples = 0
    t0 = time.perf_counter()
    for _ in range(num_batches):
        with torch.no_grad():
            inp = input_fn()
            if isinstance(inp, torch.Tensor):
                inp = inp.to(device)
            _ = model(inp)
            total_samples += batch_size
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
    return {
        "batch_size": batch_size,
        "num_batches": num_batches,
        "total_samples": total_samples,
        "elapsed_seconds": round(elapsed, 4),
        "samples_per_second": round(samples_per_sec, 2),
        "ms_per_sample": round(1000 / samples_per_sec, 4) if samples_per_sec > 0 else 0,
        "ms_per_batch": round(elapsed / num_batches * 1000, 4),
    }


# ============================================================
# Memory usage
# ============================================================


def measure_memory_usage(
    model: nn.Module,
    input_fn,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Measure peak memory usage during inference.

    CPU: psutil RSS (resident set size) + VMS
    GPU: torch.cuda.max_memory_allocated + pynvml (if available)
    """
    if device is None:
        device = next(model.parameters()).device

    result: dict[str, Any] = {}

    # CPU memory via psutil
    if _has_psutil():
        import psutil

        process = psutil.Process()
        mem_before = process.memory_info()
        with torch.no_grad():
            inp = input_fn()
            if isinstance(inp, torch.Tensor):
                inp = inp.to(device)
            _ = model(inp)
        mem_after = process.memory_info()
        result["cpu_rss_mb"] = round((mem_after.rss - mem_before.rss) / (1024 * 1024), 4)
        result["cpu_vms_mb"] = round((mem_after.vms - mem_before.vms) / (1024 * 1024), 4)
        result["cpu_rss_method"] = "psutil"
    else:
        total = sum(p.numel() for p in model.parameters())
        result["cpu_rss_mb"] = round(total * 4 / (1024 * 1024) * 3, 4)
        result["cpu_rss_method"] = "estimated (psutil not installed)"

    # GPU memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            inp = input_fn()
            if isinstance(inp, torch.Tensor):
                inp = inp.to(device)
            _ = model(inp)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result["gpu_allocated_mb"] = round(peak_allocated / (1024 * 1024), 4)
        result["gpu_reserved_mb"] = round(peak_reserved / (1024 * 1024), 4)
        result["gpu_memory_method"] = "torch.cuda.max_memory_allocated"

        # Additional pynvml metrics
        gpu_info = get_gpu_info_pynvml(device.index if device.index else 0)
        if gpu_info:
            result["gpu_pynvml"] = gpu_info

    return result


# ============================================================
# GPU monitoring via pynvml
# ============================================================


def get_gpu_info_pynvml(gpu_index: int = 0) -> dict[str, Any] | None:
    """Get detailed GPU metrics via pynvml (nvidia-ml-py).

    Returns: utilization%, memory used/total, temperature, power draw,
             clock speeds, fan speed.

    Returns None if pynvml is not installed or GPU is unavailable.
    """
    if not _has_pynvml():
        return None

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        info = {}
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            info["gpu_util_pct"] = util.gpu
            info["mem_util_pct"] = util.memory
        except Exception:
            pass

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["mem_used_mb"] = round(mem.used / (1024 * 1024), 2)
            info["mem_total_mb"] = round(mem.total / (1024 * 1024), 2)
            info["mem_free_mb"] = round(mem.free / (1024 * 1024), 2)
        except Exception:
            pass

        try:
            info["temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass

        try:
            info["power_w"] = round(
                pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 2
            )
        except Exception:
            pass

        try:
            info["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_SM
            )
            info["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_MEM
            )
        except Exception:
            pass

        try:
            info["fan_speed_pct"] = pynvml.nvmlDeviceGetFanSpeed(handle)
        except Exception:
            pass

        try:
            info["gpu_name"] = pynvml.nvmlDeviceGetName(handle).decode()
        except Exception:
            pass

        pynvml.nvmlShutdown()
        return info
    except Exception:
        return None


def measure_gpu_power_during_inference(
    model: nn.Module,
    input_fn,
    duration_seconds: float = 5.0,
    device: torch.device | None = None,
) -> dict[str, Any] | None:
    """Measure GPU power draw during continuous inference.

    Runs inference in a loop for `duration_seconds` and samples
    power draw via pynvml. Returns average/peak power.

    Returns None if pynvml is not available.
    """
    if not _has_pynvml():
        return None

    if device is None:
        device = next(model.parameters()).device
    if device.type != "cuda":
        return None

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device.index if device.index else 0)

        model.eval()
        power_samples: list[float] = []

        t_start = time.perf_counter()
        while time.perf_counter() - t_start < duration_seconds:
            with torch.no_grad():
                inp = input_fn()
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(device)
                _ = model(inp)
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_samples.append(power_w)
            except Exception:
                pass

        pynvml.nvmlShutdown()

        if not power_samples:
            return None

        arr = np.array(power_samples)
        return {
            "duration_seconds": duration_seconds,
            "num_samples": len(power_samples),
            "avg_power_w": round(float(arr.mean()), 2),
            "peak_power_w": round(float(arr.max()), 2),
            "min_power_w": round(float(arr.min()), 2),
            "samples_per_second": round(len(power_samples) / duration_seconds, 1),
            "method": "pynvml",
        }
    except Exception:
        return None


# ============================================================
# Model size on disk
# ============================================================


def measure_model_size_disk(model: nn.Module) -> dict[str, Any]:
    """Save model state_dict to temp file and measure file size."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_bytes = os.path.getsize(f.name)
        os.unlink(f.name)

    total = sum(p.numel() for p in model.parameters())
    return {
        "disk_size_bytes": size_bytes,
        "disk_size_mb": round(size_bytes / (1024 * 1024), 4),
        "disk_size_kb": round(size_bytes / 1024, 2),
        "fp32_theoretical_mb": round(total * 4 / (1024 * 1024), 4),
    }


# ============================================================
# Model comparison metrics
# ============================================================


def compute_model_mse(
    model_a: nn.Module,
    model_b: nn.Module,
    input_fn,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Compute MSE/MAE/cosine-similarity between two model outputs.

    Also computes KL divergence if outputs are log-probability-like.
    """
    if device is None:
        device = next(model_a.parameters()).device

    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        inp = input_fn()
        if isinstance(inp, torch.Tensor):
            inp = inp.to(device)
        out_a = model_a(inp)
        out_b = model_b(inp)

    # Basic metrics
    mse = float(F.mse_loss(out_a, out_b).item())
    mae = float(F.l1_loss(out_a, out_b).item())
    max_diff = float((out_a - out_b).abs().max().item())

    # Cosine similarity
    flat_a = out_a.flatten().unsqueeze(0)
    flat_b = out_b.flatten().unsqueeze(0)
    cos_sim = float(F.cosine_similarity(flat_a, flat_b).item())

    # Signal-to-Noise Ratio
    signal_power = float((out_a ** 2).mean().item())
    noise_power = mse
    snr_db = float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else float("inf")

    # KL divergence (convert to log-softmax if multi-class output)
    kl_div = None
    try:
        if out_a.shape[-1] > 1 and out_a.dim() >= 2:
            log_probs_a = F.log_softmax(out_a, dim=-1)
            probs_b = F.softmax(out_b, dim=-1)
            kl_div = float(F.kl_div(log_probs_a, probs_b, reduction="batchmean").item())
    except Exception:
        pass

    result: dict[str, Any] = {
        "mse": mse,
        "mae": mae,
        "max_abs_diff": max_diff,
        "cosine_similarity": cos_sim,
        "snr_db": snr_db,
    }
    if kl_div is not None:
        result["kl_divergence"] = kl_div

    return result


# ============================================================
# PyTorch profiler integration
# ============================================================


def profile_model(
    model: nn.Module,
    input_fn,
    output_dir: str = "./profiler_logs",
    device: torch.device | None = None,
    wait: int = 2,
    warmup: int = 2,
    active: int = 5,
    repeat: int = 2,
) -> dict[str, Any] | None:
    """Run torch.profiler on the model and return summary metrics.

    Saves Chrome trace to output_dir for visualization in chrome://tracing.

    Returns a summary dict with:
        - cpu_time_total, cuda_time_total
        - self_cpu_time_total, self_cuda_time_total
        - top_kernels_by_time

    If CUDA is not available, only CPU metrics are reported.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(wait + warmup + active):
            with torch.no_grad():
                inp = input_fn()
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(device)
                _ = model(inp)
            prof.step()

    # Extract summary
    try:
        events = prof.key_averages()
        total_cpu_time = sum(e.cpu_time_total for e in events) / 1e6  # seconds
        total_cuda_time = (
            sum(e.cuda_time_total for e in events) / 1e6
            if device.type == "cuda"
            else 0
        )

        # Top kernels
        sorted_events = sorted(events, key=lambda e: e.cpu_time_total, reverse=True)
        top_kernels = [
            {
                "name": e.key,
                "cpu_time_ms": round(e.cpu_time_total / 1000, 4),
                "cuda_time_ms": round(e.cuda_time_total / 1000, 4) if device.type == "cuda" else 0,
                "count": e.count,
                "flops": e.flops,
            }
            for e in sorted_events[:10]
        ]

        return {
            "total_cpu_time_s": round(total_cpu_time, 4),
            "total_cuda_time_s": round(total_cuda_time, 4),
            "num_events": len(events),
            "top_kernels": top_kernels,
            "trace_dir": output_dir,
            "method": "torch.profiler",
        }
    except Exception:
        return None


# ============================================================
# Logging helpers (TensorBoard / W&B)
# ============================================================


class MetricsLogger:
    """Unified logger for TensorBoard and/or W&B.

    Usage:
        logger = MetricsLogger(tb_dir="./tb_logs", project="model-compression")
        logger.log_scalar("latency_ms", 1.23, step=0)
        logger.log_metrics({"params": 1.5, "flops": 100}, step=0)
        logger.close()
    """

    def __init__(
        self,
        tb_dir: str | None = None,
        wandb_project: str | None = None,
        wandb_name: str | None = None,
        wandb_config: dict | None = None,
    ) -> None:
        self.tb_writer = None
        self.wandb_run = None
        self._active = False

        if tb_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                self._active = True
                print(f"[MetricsLogger] TensorBoard logging to {tb_dir}")
            except ImportError:
                print("[MetricsLogger] tensorboard not installed, skipping TB logging")
            except Exception as e:
                print(f"[MetricsLogger] Failed to init TensorBoard: {e}")

        if wandb_project:
            if _has_wandb():
                import wandb

                run_name = wandb_name or f"compression-{int(time.time())}"
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=wandb_config or {},
                    reinit=True,
                )
                self._active = True
                print(f"[MetricsLogger] W&B logging to {wandb_project}/{run_name}")
            else:
                print("[MetricsLogger] wandb not installed, skipping W&B logging")

    def log_scalar(self, tag: str, value: float, step: int = 0) -> None:
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        if self.wandb_run:
            self.wandb_run.log({tag: value}, step=step)

    def log_metrics(self, metrics: dict[str, Any], step: int = 0) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.log_scalar(k, v, step)

    def close(self) -> None:
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
