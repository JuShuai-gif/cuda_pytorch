"""Deterministic micro-workloads with recognizable profiler signatures.

These workloads are deliberately small and interpretable.  They are not meant
to model the performance of a real model.  Their job is to make one bottleneck
signature at a time visible in a CPU/GPU timeline.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Dict, Iterator

import torch


CASES = ("launch", "memory", "compute", "sync", "cpu")
VARIANTS = ("baseline", "optimized")


@dataclass(frozen=True)
class WorkloadConfig:
    """Shape controls shared by the CLI tools.

    Defaults are intentionally light enough for a developer workstation.  Use
    the CLI flags to scale them up after checking available memory.
    """

    numel: int = 262_144
    matrix_size: int = 384
    repeats: int = 16
    cpu_gap_ms: float = 2.0
    dtype: torch.dtype = torch.float32

    def validate(self) -> None:
        if self.numel <= 0:
            raise ValueError("numel must be positive")
        if self.matrix_size <= 0:
            raise ValueError("matrix_size must be positive")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")
        if self.cpu_gap_ms < 0:
            raise ValueError("cpu_gap_ms must be non-negative")


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` without silently falling back for explicit CUDA."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def make_inputs(
    case: str,
    config: WorkloadConfig,
    device: torch.device,
    seed: int = 2026,
) -> Dict[str, torch.Tensor]:
    """Create initialized inputs so NCU replay never reads uninitialized data."""

    config.validate()
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; expected one of {CASES}")

    # A CPU generator makes initialization deterministic across CPU and CUDA.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if case == "compute":
        n = config.matrix_size
        a = torch.randn((n, n), generator=generator, dtype=config.dtype)
        b = torch.randn((n, n), generator=generator, dtype=config.dtype)
        return {"a": a.to(device), "b": b.to(device)}

    x = torch.randn(config.numel, generator=generator, dtype=config.dtype)
    inputs = {"x": x.to(device)}
    if case == "memory":
        bias = torch.randn(config.numel, generator=generator, dtype=config.dtype)
        inputs["bias"] = bias.to(device)
    return inputs


@contextlib.contextmanager
def nvtx_range(name: str, enabled: bool, device: torch.device) -> Iterator[None]:
    """Push an NVTX push/pop range only when CUDA and NVTX are available."""

    pushed = False
    if enabled and device.type == "cuda":
        torch.cuda.nvtx.range_push(name)
        pushed = True
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


def _launch_baseline(x: torch.Tensor, repeats: int) -> torch.Tensor:
    y = x
    # Intentionally launch many tiny dependent elementwise kernels.
    for _ in range(repeats):
        y = y + 1.0
    return y


def _launch_optimized(x: torch.Tensor, repeats: int) -> torch.Tensor:
    # Same mathematical result, one elementwise kernel.
    return x + float(repeats)


def _memory_baseline(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Two complete tensor passes and an intermediate allocation.
    return x.mul(1.25).add(bias)


def _memory_optimized(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # torch.add(input, other, alpha=...) expresses the same affine operation in
    # one TensorIterator kernel and removes one global-memory round trip.
    return torch.add(bias, x, alpha=1.25)


def _compute_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Deliberate repeated GEMM: an easy-to-recognize compute-heavy baseline.
    return (a @ b) + (a @ b)


def _compute_optimized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Common-subexpression elimination: one GEMM plus one elementwise scale.
    return (a @ b) * 2.0


def _sync_baseline(x: torch.Tensor, repeats: int) -> torch.Tensor:
    y = x
    for _ in range(repeats):
        y = y + 1.0
        if y.device.type == "cuda":
            # Deliberate anti-pattern: serialize host submission and device work.
            torch.cuda.synchronize(y.device)
    return y


def _sync_optimized(x: torch.Tensor, repeats: int) -> torch.Tensor:
    y = x
    for _ in range(repeats):
        y = y + 1.0
    return y


def _cpu_baseline(x: torch.Tensor, cpu_gap_ms: float) -> torch.Tensor:
    # Synthetic input/decode/control-plane delay.  NVTX makes the gap explicit.
    time.sleep(cpu_gap_ms / 1_000.0)
    return torch.relu(x)


def _cpu_optimized(x: torch.Tensor, _cpu_gap_ms: float) -> torch.Tensor:
    return torch.relu(x)


def run_workload(
    case: str,
    variant: str,
    inputs: Dict[str, torch.Tensor],
    config: WorkloadConfig,
    *,
    emit_nvtx: bool = False,
) -> torch.Tensor:
    """Run one workload invocation and return its output tensor."""

    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; expected one of {CASES}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    device = next(iter(inputs.values())).device
    label = f"workload_{case}_{variant}"
    with nvtx_range(label, emit_nvtx, device):
        if case == "launch":
            fn = _launch_baseline if variant == "baseline" else _launch_optimized
            return fn(inputs["x"], config.repeats)
        if case == "memory":
            fn = _memory_baseline if variant == "baseline" else _memory_optimized
            return fn(inputs["x"], inputs["bias"])
        if case == "compute":
            fn = _compute_baseline if variant == "baseline" else _compute_optimized
            return fn(inputs["a"], inputs["b"])
        if case == "sync":
            fn = _sync_baseline if variant == "baseline" else _sync_optimized
            return fn(inputs["x"], config.repeats)
        fn = _cpu_baseline if variant == "baseline" else _cpu_optimized
        return fn(inputs["x"], config.cpu_gap_ms)


def estimated_executed_flops(case: str, variant: str, config: WorkloadConfig) -> int:
    """Return a documented operation-count estimate, not a hardware counter.

    The value intentionally describes executed arithmetic in these source-level
    algorithms.  It must not be presented as measured FLOPs or MFU.
    """

    if case in ("launch", "sync"):
        return config.numel * (config.repeats if variant == "baseline" else 1)
    if case == "memory":
        return config.numel * 2
    if case == "compute":
        gemm = 2 * config.matrix_size**3
        return (2 * gemm + config.matrix_size**2) if variant == "baseline" else (gemm + config.matrix_size**2)
    return config.numel  # relu comparison, approximate
