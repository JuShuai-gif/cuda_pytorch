"""Workload construction, analytical cost estimates, and correctness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from . import baseline, optimized
from .common import dtype_nbytes


WORKLOAD_NAMES = ("launch", "memory", "sync", "cpu_gap", "gemm")


@dataclass(frozen=True)
class CostEstimate:
    flops_per_call: int
    bytes_per_call_lower_bound: int
    effective_elements_per_call: int
    caveat: str


@dataclass(frozen=True)
class PreparedWorkload:
    name: str
    expected_bottleneck: str
    baseline: Callable[[], Any]
    optimized: Callable[[], Any]
    costs: dict[str, CostEstimate]
    config: dict[str, Any]


def prepare_workload(
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    vector_elements: int,
    inner_iterations: int,
    matrix_size: int,
    cpu_delay_ms: float,
) -> PreparedWorkload:
    if name not in WORKLOAD_NAMES:
        raise ValueError(f"Unknown workload {name!r}; choose from {WORKLOAD_NAMES}")
    if vector_elements < 1 or inner_iterations < 1 or matrix_size < 1 or cpu_delay_ms < 0:
        raise ValueError("sizes/iterations must be positive and cpu_delay_ms must be non-negative")

    element_bytes = dtype_nbytes(dtype)
    if name == "launch":
        x = torch.randn(vector_elements, device=device, dtype=dtype)
        increment = 0.125
        return PreparedWorkload(
            name=name,
            expected_bottleneck="launch-bound for small tensors",
            baseline=lambda: baseline.launch_chain(x, inner_iterations, increment),
            optimized=lambda: optimized.launch_collapsed(x, inner_iterations, increment),
            costs={
                "baseline": CostEstimate(
                    vector_elements * inner_iterations,
                    2 * vector_elements * element_bytes * inner_iterations,
                    vector_elements,
                    "Assumes every eager add reaches device memory; cache can reduce HBM traffic.",
                ),
                "optimized": CostEstimate(
                    vector_elements,
                    2 * vector_elements * element_bytes,
                    vector_elements,
                    "Counts the scalar multiplication as host-side constant folding.",
                ),
            },
            config={"vector_elements": vector_elements, "inner_iterations": inner_iterations},
        )

    if name == "memory":
        x = torch.randn(vector_elements, device=device, dtype=dtype)
        return PreparedWorkload(
            name=name,
            expected_bottleneck="memory-bandwidth/allocation-bound",
            baseline=lambda: baseline.redundant_copies(x, inner_iterations),
            optimized=lambda: optimized.single_copy(x, inner_iterations),
            costs={
                "baseline": CostEstimate(
                    0,
                    2 * vector_elements * element_bytes * inner_iterations,
                    vector_elements,
                    "Clone reads and writes the tensor; allocator metadata traffic is excluded.",
                ),
                "optimized": CostEstimate(
                    0,
                    2 * vector_elements * element_bytes,
                    vector_elements,
                    "One observable clone remains.",
                ),
            },
            config={"vector_elements": vector_elements, "copies": inner_iterations},
        )

    if name == "sync":
        x = torch.randn(vector_elements, device=device, dtype=dtype)
        scale, bias = 1.0001, 0.0001
        approximate_flops = 3 * vector_elements * inner_iterations
        approximate_bytes = 5 * vector_elements * element_bytes * inner_iterations
        caveat = (
            "Approximation assumes separate eager mul/add/reduction kernels; profiler is authoritative."
        )
        return PreparedWorkload(
            name=name,
            expected_bottleneck="host/device synchronization-bound",
            baseline=lambda: baseline.synchronize_each_iteration(
                x, inner_iterations, scale, bias
            ),
            optimized=lambda: optimized.synchronize_once(x, inner_iterations, scale, bias),
            costs={
                "baseline": CostEstimate(
                    approximate_flops, approximate_bytes, vector_elements, caveat
                ),
                "optimized": CostEstimate(
                    approximate_flops + inner_iterations,
                    approximate_bytes,
                    vector_elements,
                    caveat + " Stack/final reduction traffic is omitted.",
                ),
            },
            config={"vector_elements": vector_elements, "inner_iterations": inner_iterations},
        )

    if name == "cpu_gap":
        x = torch.randn(vector_elements, device=device, dtype=dtype)
        delay_seconds = cpu_delay_ms / 1_000.0
        cost = CostEstimate(
            vector_elements,
            2 * vector_elements * element_bytes,
            vector_elements,
            "The intentional CPU sleep has no FLOPs or device bytes.",
        )
        return PreparedWorkload(
            name=name,
            expected_bottleneck="CPU/input-pipeline-bound",
            baseline=lambda: baseline.cpu_gap(x, delay_seconds),
            optimized=lambda: optimized.cpu_ready(x, delay_seconds),
            costs={"baseline": cost, "optimized": cost},
            config={"vector_elements": vector_elements, "cpu_delay_ms": cpu_delay_ms},
        )

    a = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    flops = 2 * matrix_size**3
    minimum_bytes = 3 * matrix_size**2 * element_bytes
    cost = CostEstimate(
        flops,
        minimum_bytes,
        matrix_size,
        "Algorithmic minimum assumes each input is read and output written once; actual hierarchy traffic differs.",
    )
    return PreparedWorkload(
        name=name,
        expected_bottleneck="compute-bound only when shape/dtype/hardware reach GEMM saturation",
        baseline=lambda: baseline.gemm_einsum(a, b),
        optimized=lambda: optimized.gemm_mm(a, b),
        costs={"baseline": cost, "optimized": cost},
        config={"matrix_size": matrix_size},
    )


def _tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().to(dtype=torch.float64, device="cpu")
    got = candidate.detach().to(dtype=torch.float64, device="cpu")
    absolute = (ref - got).abs()
    denominator = ref.abs().clamp_min(1e-12)
    return {
        "max_abs_error": float(absolute.max().item()) if absolute.numel() else 0.0,
        "max_rel_error": float((absolute / denominator).max().item()) if absolute.numel() else 0.0,
    }


def compare_outputs(
    reference: Any,
    candidate: Any,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if isinstance(reference, tuple):
        if not isinstance(candidate, tuple) or len(reference) != len(candidate):
            return {"passed": False, "reason": "output structures differ"}
        details = [
            compare_outputs(left, right, rtol=rtol, atol=atol)
            for left, right in zip(reference, candidate)
        ]
        return {"passed": all(item["passed"] for item in details), "components": details}
    if isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor):
        errors = _tensor_error(reference, candidate)
        return {
            "passed": bool(torch.allclose(reference, candidate, rtol=rtol, atol=atol)),
            **errors,
        }
    left, right = float(reference), float(candidate)
    difference = abs(left - right)
    threshold = atol + rtol * abs(left)
    return {
        "passed": difference <= threshold,
        "abs_error": difference,
        "tolerance": threshold,
    }

